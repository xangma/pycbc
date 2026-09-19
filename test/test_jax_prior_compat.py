# Copyright (C) 2026 PyCBC developers
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused JAX compatibility tests for inference prior distributions."""

import numpy
import pytest

from pycbc import distributions, transforms

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)


def _assert_matches(result, expected, *, rtol=1e-11, atol=1e-12):
    assert isinstance(result, (jnp.ndarray, jax.Array))
    numpy.testing.assert_allclose(
        numpy.asarray(result), expected, rtol=rtol, atol=atol,
        equal_nan=True
    )


def test_common_and_joint_priors_preserve_tensors():
    dtype = jnp.float64
    uniform = distributions.Uniform(u=(-2.0, 2.0), v=(0.0, 1.0))
    gaussian = distributions.Gaussian(
        x=(-2.0, 2.0), x_mean=0.25, x_var=1.5
    )
    power = distributions.UniformPowerLaw(dim=3, r=(1.0, 4.0))
    log_uniform = distributions.UniformLog10(scale=(1.0, 100.0))
    joint = distributions.JointDistribution(("x", "r"), gaussian, power)
    joint._constraints = [
        distributions.constraints.Constraint("(x >= -1.0) & (r < 3.0)")
    ]

    x_values = numpy.array([-1.5, -0.25, 0.75])
    r_values = numpy.array([1.1, 2.0, 3.5])
    scale_values = numpy.array([1.0, 10.0, 99.0])
    u_values = numpy.array([-3.0, 0.0, 1.0])
    v_values = numpy.array([0.5, 0.5, 1.0])
    expected_gaussian = numpy.array(
        [gaussian.logpdf(x=value) for value in x_values]
    )
    expected_power = numpy.array(
        [power.logpdf(r=value) for value in r_values]
    )
    expected_scale = numpy.array(
        [log_uniform.logpdf(scale=value) for value in scale_values]
    )
    expected_uniform = numpy.array([
        uniform.logpdf(u=u, v=v) for u, v in zip(u_values, v_values)
    ])

    x = jnp.array(x_values, dtype=dtype)
    radius = jnp.array(r_values, dtype=dtype)
    scale = jnp.array(scale_values, dtype=dtype)
    gaussian_log = gaussian.logpdf(x=x)
    power_log = power.logpdf(r=radius)
    scale_log = log_uniform.logpdf(scale=scale)
    uniform_log = uniform.logpdf(
        u=jnp.array(u_values, dtype=dtype),
        v=jnp.array(v_values, dtype=dtype),
    )
    joint_log = joint(x=x, r=radius)
    contains = joint.contains({"x": x, "r": radius})
    constrained = joint.within_constraints({"x": x, "r": radius})

    _assert_matches(gaussian_log, expected_gaussian)
    _assert_matches(power_log, expected_power)
    _assert_matches(scale_log, expected_scale)
    _assert_matches(uniform_log, expected_uniform)
    assert contains.dtype == jnp.bool_
    assert contains.tolist() == [False, True, False]
    assert constrained.dtype == jnp.bool_
    assert constrained.tolist() == [False, True, False]
    assert bool(jnp.array_equal(contains, constrained))
    assert bool(jnp.array_equal(jnp.isfinite(joint_log), constrained))

    def grad_loss(xv, rv, sv):
        gl = gaussian.logpdf(x=xv)
        pl = power.logpdf(r=rv)
        sl = log_uniform.logpdf(scale=sv)
        return jnp.sum(gl + pl + sl)

    gx, gr, gs = jax.grad(grad_loss, argnums=(0, 1, 2))(x, radius, scale)
    numpy.testing.assert_allclose(gx, -(x_values - 0.25) / 1.5, rtol=1e-12)
    numpy.testing.assert_allclose(gr, 2.0 / r_values, rtol=1e-12)
    numpy.testing.assert_allclose(gs, -1.0 / scale_values, rtol=1e-12)


def test_transformed_and_angular_constraints_preserve_boolean_tensors():
    dtype = jnp.float64
    shifted = transforms.CustomTransform(
        ["x"], ["shifted_x"], {"shifted_x": "x + 0.5"}
    )
    transformed = distributions.constraints.Constraint(
        "shifted_x < 1.0", transforms=[shifted]
    )
    angular = distributions.constraints.Constraint(
        "(dec + ddec >= -pi/2) & (dec + ddec <= pi/2)"
    )
    x = jnp.array([-1.5, -0.25, 0.75], dtype=dtype)
    dec = jnp.array([-1.4, 0.0, 1.4], dtype=dtype)
    ddec = jnp.array([-0.3, 0.2, 0.3], dtype=dtype)

    transformed_result = transformed({"x": x})
    angular_result = angular({"dec": dec, "ddec": ddec})

    assert isinstance(transformed_result, (jnp.ndarray, jax.Array))
    assert transformed_result.dtype == jnp.bool_
    assert transformed_result.tolist() == [True, True, False]

    assert isinstance(angular_result, (jnp.ndarray, jax.Array))
    assert angular_result.dtype == jnp.bool_
    assert angular_result.tolist() == [False, True, False]


def test_angular_and_solid_angle_priors_preserve_gradients():
    dtype = jnp.float64
    sin_prior = distributions.SinAngle(theta=(0.2, 2.9))
    cos_prior = distributions.CosAngle(declination=(-1.2, 1.2))
    solid_prior = distributions.UniformSolidAngle(
        polar_bounds=(0.3, 2.8), azimuthal_bounds=(0.1, 5.9)
    )

    theta_values = numpy.array([0.4, 1.2, 2.5])
    declination_values = numpy.array([-0.8, 0.1, 0.9])
    unit_values = numpy.array([0.15, 0.5, 0.85])
    azimuthal_values = numpy.array([0.2, 0.45, 0.9])

    expected_sin = numpy.array([sin_prior.logpdf(theta=v) for v in theta_values])
    expected_cos = numpy.array(
        [cos_prior.logpdf(declination=v) for v in declination_values]
    )
    expected_solid = solid_prior.cdfinv(
        theta=unit_values, phi=azimuthal_values
    )

    theta = jnp.array(theta_values, dtype=dtype)
    declination = jnp.array(declination_values, dtype=dtype)
    unit = jnp.array(unit_values, dtype=dtype)
    azimuthal = jnp.array(azimuthal_values, dtype=dtype)

    sin_log = sin_prior.logpdf(theta=theta)
    cos_log = cos_prior.logpdf(declination=declination)
    solid = solid_prior.cdfinv(theta=unit, phi=azimuthal)

    _assert_matches(sin_log, expected_sin)
    _assert_matches(cos_log, expected_cos)
    _assert_matches(solid["theta"], expected_solid["theta"])
    _assert_matches(solid["phi"], expected_solid["phi"])

    def prior_loss(th, dec, u, az):
        sl = sin_prior.logpdf(theta=th)
        cl = cos_prior.logpdf(declination=dec)
        sd = solid_prior.cdfinv(theta=u, phi=az)
        return jnp.sum(sl + cl + sd["theta"] + sd["phi"])

    grads = jax.grad(prior_loss, argnums=(0, 1, 2, 3))(theta, declination, unit, azimuthal)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


def test_kde_and_spin_priors_preserve_tensors():
    dtype = jnp.float64
    rng = numpy.random.default_rng(1984)
    samples_x = rng.normal(size=64)
    samples_y = numpy.clip(
        0.5 + 0.14 * samples_x + 0.1 * rng.normal(size=64), 0.04, 0.96
    )
    prior = distributions.Arbitrary(
        bounds={"y": (0.0, 1.0)}, x=samples_x, y=samples_y
    )
    x_values = numpy.array([-1.0, 0.25, 1.4])[:, None]
    y_values = numpy.array([0.2, 0.7])[None, :]
    expected = numpy.array([
        [prior.logpdf(x=float(x), y=float(y)) for y in y_values[0]]
        for x in x_values[:, 0]
    ])
    x = jnp.array(x_values, dtype=dtype)
    y = jnp.array(y_values, dtype=dtype)

    actual = prior.logpdf(x=x, y=y)

    _assert_matches(actual, expected, rtol=1e-10)

    def kde_loss(xv, yv):
        return jnp.sum(prior.logpdf(x=xv, y=yv))

    gx, gy = jax.grad(kde_loss, argnums=(0, 1))(x, y)
    assert bool(jnp.all(jnp.isfinite(gx)))
    assert bool(jnp.all(jnp.isfinite(gy)))
    assert prior._jax_kde_cache
    prior.set_bandwidth("silverman")
    assert not prior._jax_kde_cache

    spin_prior = distributions.IndependentChiPChiEff(
        mass1=(10.0, 20.0), mass2=(5.0, 10.0), nsamples=64, seed=17
    )
    spin_values = {
        "mass1": [15.0, 25.0],
        "mass2": [8.0, 8.0],
        "xi1": [0.2, 0.2],
        "xi2": [0.1, 0.1],
        "chi_eff": [0.0, 0.0],
        "chi_a": [0.0, 0.0],
        "phi_a": [1.0, 1.0],
        "phi_s": [1.5, 1.5],
    }
    spin_tensors = {
        name: jnp.array(values, dtype=dtype)
        for name, values in spin_values.items()
    }
    spin_contains = spin_prior.__contains__(spin_tensors)
    assert isinstance(spin_contains, (jnp.ndarray, jax.Array))
    assert spin_contains.dtype == jnp.bool_
    assert spin_contains.tolist() == [True, False]


def test_tabulated_prior_interpolation_preserves_gradients(tmp_path):
    dtype = jnp.float64
    knots = numpy.array([-2.0, 0.0, 1.0, 3.0])
    density = numpy.array([1.0, 3.0, 2.0, 1.0])
    density_file = tmp_path / "tabulated-prior.txt"
    numpy.savetxt(density_file, numpy.column_stack((knots, density)))
    prior = distributions.DistributionFunctionFromFile(
        params=["x"], file_path=density_file, column_index=1
    )
    values = numpy.array([-1.0, 0.5, 2.0])
    units = numpy.array([0.09, 0.41, 0.83])
    expected_pdf = numpy.array([prior._pdf(value) for value in values])
    expected_cdf = prior._cdf(values)
    expected_inverse = prior.cdfinv(x=units)["x"]
    value_tensor = jnp.array(values, dtype=dtype)
    unit_tensor = jnp.array(units, dtype=dtype)

    actual_pdf = prior._pdf(value_tensor)
    actual_cdf = prior._cdf(value_tensor)
    actual_inverse = prior.cdfinv(x=unit_tensor)["x"]

    _assert_matches(actual_pdf, expected_pdf)
    _assert_matches(actual_cdf, expected_cdf)
    _assert_matches(actual_inverse, expected_inverse)

    def tab_loss(vt, ut):
        p = prior._pdf(vt)
        c = prior._cdf(vt)
        inv = prior.cdfinv(x=ut)["x"]
        return jnp.sum(p + c + inv)

    gv, gu = jax.grad(tab_loss, argnums=(0, 1))(value_tensor, unit_tensor)
    assert bool(jnp.all(jnp.isfinite(gv)))
    assert bool(jnp.all(jnp.isfinite(gu)))


def test_fixed_samples_preserve_pairing_and_device():
    dtype = jnp.float64
    samples = {
        "x": numpy.array([-4.0, -2.0, -1.0, 0.5, 1.5, 3.0]),
        "y": numpy.array([9.0, -3.0, 8.0, 1.0, -4.0, 6.0]),
    }
    unit_x = jnp.array(
        [0.0, 0.18, 0.45, 0.72, 1.0], dtype=dtype
    )
    unit_y = jnp.array(
        [0.95, 0.05, 0.51, 0.35, 1.0], dtype=dtype
    )
    tensor_samples = {
        name: jnp.array(values, dtype=dtype)
        for name, values in samples.items()
    }
    prior = distributions.FixedSamples(("x", "y"), tensor_samples)

    inverse = prior.cdfinv(x=unit_x, y=unit_y)
    draws = prior.rvs(size=32)

    for result in (inverse, draws):
        assert all(isinstance(value, (jnp.ndarray, jax.Array)) for value in result.values())
        assert all(value.dtype == dtype for value in result.values())
    paired = (
        (draws["x"][:, None] == tensor_samples["x"][None, :])
        & (draws["y"][:, None] == tensor_samples["y"][None, :])
    )
    assert bool(jnp.all(jnp.any(paired, axis=1)))


def test_convex_hull_constraint_preserves_boolean_tensor():
    hull_points = numpy.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ])
    values = {
        "coeff_0": numpy.array([0.1, 0.9, 1.1, -0.1, numpy.nan]),
        "coeff_1": numpy.array([0.1, 0.9, 0.6, 0.2, 0.1]),
        "coeff_2": numpy.array([0.1, 0.9, 0.1, 0.2, 0.1]),
    }
    constraint = object.__new__(
        distributions.constraints.SupernovaeConvexHull
    )
    distributions.constraints.Constraint.__init__(constraint, "unused")
    constraint.hull_dimention = 3
    constraint.required_parameters = ["coeff_0", "coeff_1", "coeff_2"]
    constraint._hull = distributions.constraints.scipy.spatial.Delaunay(
        hull_points
    )
    constraint._jax_hull_cache = {}
    expected = constraint(values)
    tensor_values = {
        name: jnp.array(value, dtype=jnp.float64)
        for name, value in values.items()
    }

    actual = constraint(tensor_values)

    assert isinstance(actual, (jnp.ndarray, jax.Array))
    assert actual.dtype == jnp.bool_
    numpy.testing.assert_array_equal(actual, expected)
    assert len(constraint._jax_hull_cache) == 1


def test_mass_ratio_and_qnm_priors_accept_tensor_batches():
    dtype = jnp.float64
    mass_prior = distributions.QfromUniformMass1Mass2(q=(1.0, 8.0))
    q_values = numpy.array([1.0, 2.5, 7.5])
    expected = numpy.array(
        [mass_prior.logpdf(q=value) for value in q_values]
    )
    q = jnp.array(q_values, dtype=dtype)

    logpdf = mass_prior.logpdf(q=q)

    _assert_matches(logpdf, expected)

    def q_loss(qv):
        return jnp.sum(mass_prior.logpdf(q=qv))

    gq = jax.grad(q_loss)(q)
    assert bool(jnp.all(jnp.isfinite(gq)))

    qnm_prior = distributions.UniformF0Tau(
        f0=(100.0, 500.0), tau=(0.001, 0.02), norm_tolerance=0.1
    )
    f0 = jnp.array([200.0, 80.0], dtype=dtype)
    tau = jnp.array([0.004, 0.004], dtype=dtype)
    contained = qnm_prior.__contains__({"f0": f0, "tau": tau})
    assert isinstance(contained, (jnp.ndarray, jax.Array))
    assert contained.dtype == jnp.bool_
    assert contained.tolist() == [True, False]


def test_custom_constraint_accepts_scalar_math_and_static_arguments():
    constraint = distributions.constraints.Constraint(
        "maximum(x, 0) < sqrt(limit)", static_args={"limit": 2.0}
    )
    x = jnp.array([-1.0, 0.5, 1.5], dtype=jnp.float64)
    result = constraint({"x": x})
    assert isinstance(result, (jnp.ndarray, jax.Array))
    assert result.dtype == jnp.bool_
    numpy.testing.assert_array_equal(result, [True, True, False])
