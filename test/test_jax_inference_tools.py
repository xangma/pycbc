"""Focused tests for lazy JAX inference helper dispatch."""

import subprocess
import sys

import numpy
import pytest


def test_numpy_subset_uses_the_model_generator(monkeypatch):
    from pycbc.inference.models import proposals_jax as tools

    def reject_global_choice(*args, **kwargs):
        raise AssertionError("subset sampling must use the model generator")

    monkeypatch.setattr(numpy.random, "choice", reject_global_choice)
    expected = numpy.random.default_rng(27).choice(100, size=7, replace=False)
    choice, host_choice = tools._random_permutation(
        numpy.arange(100), 7, numpy.random.default_rng(27)
    )
    numpy.testing.assert_array_equal(choice, expected)
    assert host_choice is choice


def test_numpy_helpers_do_not_import_jax_backend():
    code = """
import sys
import numpy
from pycbc.inference.models import gaussian_noise, marginalized_gaussian_noise, relbin, tools
from pycbc.types import Array
assert 'pycbc.inference.models.tools_jax' not in sys.modules
assert 'jax' not in sys.modules
assert 'pycbc.inference.models.gaussian_noise_jax' not in sys.modules
assert Array(numpy.array([1., 2.])).inner(Array(numpy.array([3., 4.]))) == 11.
assert numpy.isfinite(tools.marginalize_likelihood(1., 2., phase=True))
assert 'pycbc.inference.models.tools_jax' not in sys.modules
assert 'jax' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_jax_dispatch_matches_numpy_and_preserves_gradients():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pycbc.inference.models import proposals_jax as tools
    from pycbc.inference.models import tools as native_tools

    h_numpy = numpy.array([1.0 + 2.0j, 3.0 - 1.0j])
    d_numpy = numpy.array([2.0 - 1.0j, 0.5 + 4.0j])
    h = jnp.array(h_numpy, dtype=jnp.complex128)
    d = jnp.array(d_numpy, dtype=jnp.complex128)

    got_hd, got_hh = tools._fused_inner_hd_hh(h, d)
    from pycbc.types import Array
    expected_hd = Array(h_numpy).inner(Array(d_numpy))
    expected_hh = Array(h_numpy).inner(Array(h_numpy)).real
    numpy.testing.assert_allclose(numpy.asarray(got_hd), expected_hd)
    numpy.testing.assert_allclose(numpy.asarray(got_hh), expected_hh)

    def cost(h_in):
        hd, hh = tools._fused_inner_hd_hh(h_in, d)
        return jnp.real(hd) + hh

    grad_fn = jax.grad(cost)
    grad = grad_fn(h)
    assert grad is not None
    assert grad.shape == h.shape

    loglr = numpy.array([-1.0, 0.5, 1.5, -0.25])
    numpy.random.seed(91)
    got_indices = tools.draw_sample(
        jnp.array(loglr, dtype=jnp.float64), size=16
    )
    numpy.random.seed(91)
    expected_indices = tools.draw_sample(loglr, size=16)
    numpy.testing.assert_array_equal(got_indices, expected_indices)

    sh = numpy.array([1.5 + 0.25j, 0.5 - 1.0j, 2.0 + 0.5j])
    hh = numpy.array([0.75, 1.25, 2.5])
    expected = native_tools.marginalize_likelihood(sh, hh, phase=True)
    actual = native_tools.marginalize_likelihood(
        jnp.array(sh, dtype=jnp.complex128),
        jnp.array(hh, dtype=jnp.float64),
        phase=True,
    )
    assert actual == pytest.approx(expected)


def test_public_backend_protocol_and_frequency_lookup():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from pycbc.inference.models import proposals_jax as tools

    class PublicJaxValue:
        backend = "jax"

        def __init__(self, value):
            self.backend_array = value

    left = PublicJaxValue(jnp.array([1.0, 2.0], dtype=jnp.float64))
    right = PublicJaxValue(jnp.array([3.0, 4.0], dtype=jnp.float64))
    assert float(tools._inner(left, right)) == pytest.approx(11.0)

    frequencies = jnp.arange(16, dtype=jnp.float64) * 0.25
    assert tools._last_index_at_or_below(frequencies, 1.1) == 4
    with pytest.raises(IndexError, match="no values"):
        tools._last_index_at_or_below(frequencies, -0.1)
