"""Original numerical boundaries for device-resident inference."""


import numpy as np


import pytest


jax = pytest.importorskip("jax")


import jax.numpy as jnp


from pycbc.scheme import CPUScheme, JAXScheme


from pycbc.types import Array


from pycbc.inference.models import tools_jax


from pycbc.inference.models.tools import marginalize_likelihood


@pytest.fixture(params=("cpu", "cuda"))
def inference_device(request):
    if request.param == "cuda":
        try:
            devices = jax.devices("gpu")
        except RuntimeError:
            devices = []
        if not devices:
            pytest.skip("CUDA is unavailable")
    return request.param


def _bytes(value):
    arr = np.asarray(value)
    return arr.dtype, arr.shape, arr.tobytes()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("selector", ["inner", "inference_inner"])
def test_original_inner_is_independently_selectable(dtype, selector, inference_device):
    values = np.asarray([1.234567 + 0.3j, -2.125 + 1.1j], dtype=dtype)
    other = np.asarray([0.35 - 1.4j, 2.02 + 0.27j], dtype=dtype)
    with CPUScheme():
        expected = Array(values).inner(Array(other))
    with JAXScheme(inference_device, reference_operations=(selector,)):
        actual = tools_jax.inner(jnp.asarray(values), jnp.asarray(other))
    assert _bytes(actual) == _bytes(expected)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_weighting_and_inner_controls_compose_exactly(dtype, inference_device):
    rng = np.random.default_rng(121)
    h = (rng.normal(size=33) + 1j * rng.normal(size=33)).astype(dtype)
    d = (rng.normal(size=33) + 1j * rng.normal(size=33)).astype(dtype)
    # Match the precision required by original Array arithmetic.
    w = rng.uniform(0.25, 2.0, size=33).astype(h.real.dtype)
    with CPUScheme():
        weighted = Array(h)
        weighted *= Array(w)
        expected = weighted.inner(Array(d)), weighted.inner(weighted).real
    with JAXScheme(
        inference_device, reference_operations=("inference_whitening", "inner")
    ):
        actual = tools_jax.fused_inner_hd_hh(
            jnp.asarray(h), jnp.asarray(d), jnp.asarray(w)
        )
    assert tuple(map(_bytes, actual)) == tuple(map(_bytes, expected))


@pytest.mark.parametrize("phase", [False, True])
def test_original_marginalization_is_exact(phase):
    sh = np.asarray([1.35 + 0.21j, -2.01 + 0.3j, 0.75 - 0.35j])
    hh = np.asarray([1.05, 3.11, 0.69])
    weights = np.log(np.asarray([0.2, 0.3, 0.5]))
    with CPUScheme():
        expected = marginalize_likelihood(
            sh, hh, phase=phase, logw=weights, return_peak=True
        )
    with JAXScheme(reference_operations=("inference_marginalization",)):
        actual = marginalize_likelihood(
            jnp.asarray(sh),
            jnp.asarray(hh),
            phase=phase,
            logw=weights,
            return_peak=True,
        )
    assert tuple(map(_bytes, actual)) == tuple(map(_bytes, expected))


def test_default_inner_and_marginalization_remain_differentiable():
    with JAXScheme():
        data = jnp.asarray([1.0 + 0.5j, -0.2 + 0.3j])

        def loss(amplitude):
            hd, hh = tools_jax.fused_inner_hd_hh(amplitude * data, data)
            return marginalize_likelihood(hd, hh, phase=True, skip_vector=True)

        actual = jax.grad(loss)(jnp.asarray(0.7))
        step = 1e-5
        finite = (loss(0.7 + step) - loss(0.7 - step)) / (2 * step)
        np.testing.assert_allclose(actual, finite, rtol=1e-8)


def test_original_sampling_preserves_rng_stream_and_indices():
    from pycbc.inference.models.tools import draw_sample

    logs = np.random.default_rng(11).normal(size=4096).astype(np.float32)
    before = np.random.get_state()
    try:
        np.random.seed(419)
        expected = draw_sample(logs, size=100000)
        expected_next = np.random.random()
        np.random.seed(419)
        with JAXScheme(reference_operations=("inference_sampling",)):
            actual = tools_jax.draw_sample(jnp.asarray(logs), size=100000, host=False)
        actual_next = np.random.random()
        assert _bytes(actual) == _bytes(expected)
        assert actual_next == expected_next
    finally:
        np.random.set_state(before)


def test_original_distance_scalar_and_interpolation_marginalization():
    from scipy.interpolate import RectBivariateSpline

    sh = np.complex128(1.35 + 0.21j)
    hh = np.float64(1.05)
    distance = (np.asarray([0.5, 1.0, 2.0]), np.asarray([0.2, 0.3, 0.5]))
    expected = marginalize_likelihood(sh, hh, phase=True, distance=distance)
    with JAXScheme(reference_operations=("inference_marginalization",)):
        actual = marginalize_likelihood(
            jnp.asarray(sh), jnp.asarray(hh), phase=True, distance=distance
        )
    assert _bytes(actual) == _bytes(expected)
    x = np.linspace(0.1, 5.0, 6)
    y = np.linspace(0.2, 4.0, 6)
    spline = RectBivariateSpline(x, y, np.outer(x, y))
    evaluator = tools_jax.rect_bivariate_spline_evaluator(spline)
    sh = np.asarray([0.5, 1.25, 2.5])
    hh = np.asarray([0.8, 1.0, 2.0])
    expected = marginalize_likelihood(
        sh, hh, interpolator=lambda a, b: spline(a, b, grid=False)
    )
    with JAXScheme(
        reference_operations=("inference_marginalization", "inference_interpolant")
    ):
        actual = marginalize_likelihood(
            jnp.asarray(sh), jnp.asarray(hh), interpolator=evaluator
        )
    assert _bytes(actual) == _bytes(expected)


def test_spline_gradient_before_compilation_preserves_evaluator():
    from scipy.interpolate import RectBivariateSpline

    x = np.linspace(0.1, 5.0, 6)
    y = np.linspace(0.2, 4.0, 6)
    evaluator = tools_jax.rect_bivariate_spline_evaluator(
        RectBivariateSpline(x, y, np.outer(x, y))
    )
    with JAXScheme():
        np.testing.assert_allclose(
            jax.grad(lambda z: evaluator(z, jnp.asarray(0.8)))(jnp.asarray(0.7)), 0.8
        )
        np.testing.assert_allclose(
            jax.jit(evaluator)(jnp.asarray(0.7), jnp.asarray(0.8)), 0.56
        )


def test_selected_device_survives_ambient_override_in_numerical_controls():
    import os
    import subprocess
    import sys

    code = """
import numpy as np
import jax
from scipy.interpolate import RectBivariateSpline
from pycbc.scheme import JAXScheme
from pycbc.types.array_jax import to_jax
from pycbc.inference.models.tools_jax import inner, draw_sample, rect_bivariate_spline_evaluator
x=np.linspace(.1,5.,6);y=np.linspace(.2,4.,6)
evaluate=rect_bivariate_spline_evaluator(RectBivariateSpline(x,y,np.outer(x,y)))
for controls in ((),('inner','inference_sampling','inference_interpolant')):
 with JAXScheme('cpu:1',reference_operations=controls):
  values=to_jax(np.ones(6,np.complex128))
  with jax.default_device(jax.devices('cpu')[0]):
   results=(inner(values,values),draw_sample(values.real,size=2,host=False),evaluate(values.real,values.real))
  assert all(result.device==jax.devices('cpu')[1] for result in results)
"""
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
