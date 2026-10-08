"""Device-resident proposal and marginalization support."""


from pycbc.types.backend import backend_array, is_backend


def _jax_array(value):
    """Return the public JAX backend array, if present."""
    return backend_array(value, "jax")


def _jax_tools(*values):
    """Load the JAX implementation only for JAX-backed inputs."""
    if not any(is_backend(value, "jax") for value in values):
        return None
    from pycbc.inference.models import tools_jax

    return tools_jax


def _backend_tools(*values):
    """Load the matching backend module (JAX) if present."""
    return _jax_tools(*values)


def _inner(left, right):
    """Return an inner product without scalarizing backend reductions."""
    backend = _backend_tools(left, right)
    if backend is not None and all(is_backend(value, "jax") for value in (left, right)):
        return backend.inner(left, right)
    return left.inner(right)


def _real_inner(left, right):
    """Return a real inner product without scalarizing backend reductions."""
    backend = _backend_tools(left, right)
    if backend is not None and all(is_backend(value, "jax") for value in (left, right)):
        return backend.real_inner(left, right)
    return _inner(left, right).real


def _fused_inner_hd_hh(h, d, weight=None):
    """Use the JAX implementation without adding a separate CPU algorithm."""
    from .tools_jax import fused_inner_hd_hh

    return fused_inner_hd_hh(h, d, weight)


def _marginalize_likelihood_jax(
    sh,
    hh,
    logw,
    phase,
    distance,
    skip_vector,
    return_peak,
    return_complex,
    interpolator=None,
):
    """JAX implementation of explicit likelihood marginalizations."""
    backend = _jax_tools(sh, hh)
    return backend.marginalize_likelihood(
        sh,
        hh,
        logw,
        phase,
        distance,
        skip_vector,
        return_peak,
        return_complex,
        interpolator=interpolator,
    )


def _distance_interpolant(native, spline):
    from .tools_jax import rect_bivariate_spline_evaluator

    evaluate = rect_bivariate_spline_evaluator(spline)

    def wrapper(x, y, bounds_check=True):
        if _jax_array(x) is not None or _jax_array(y) is not None:
            return evaluate(x, y, bounds_check=bounds_check)
        return native(x, y, bounds_check=bounds_check)

    wrapper._jax_evaluate = evaluate
    return wrapper
