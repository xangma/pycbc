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
        numpy.asarray(result), expected, rtol=rtol, atol=atol, equal_nan=True
    )


def test_common_and_joint_priors_preserve_tensors():
    dtype = jnp.float64
    uniform = distributions.Uniform(u=(-2.0, 2.0), v=(0.0, 1.0))
    gaussian = distributions.Gaussian(x=(-2.0, 2.0), x_mean=0.25, x_var=1.5)
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
    expected_power = numpy.array([power.logpdf(r=value) for value in r_values])
    expected_scale = numpy.array(
        [log_uniform.logpdf(scale=value) for value in scale_values]
    )
    expected_uniform = numpy.array(
        [uniform.logpdf(u=u, v=v) for u, v in zip(u_values, v_values)]
    )

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

    expected_sin = numpy.array(
        [sin_prior.logpdf(theta=v) for v in theta_values]
    )
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

    grads = jax.grad(prior_loss, argnums=(0, 1, 2, 3))(
        theta, declination, unit, azimuthal
    )
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
    # The original SciPy KDE and public coordinate transform are the oracle.
    # NumPy 2.4+ rejects the old public scalar float(length-one-array) cast.
    transform = prior._transforms[prior._tparams["y"]]
    expected = numpy.empty((len(x_values), y_values.shape[1]))
    for ix, xv in enumerate(x_values[:, 0]):
        for iy, yv in enumerate(y_values[0]):
            mapped = transform.transform({"y": yv})
            density = prior.kde.evaluate([[xv], [mapped["logity"]]])[0]
            expected[ix, iy] = numpy.log(density * transform.jacobian(mapped))
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
    unit_x = jnp.array([0.0, 0.18, 0.45, 0.72, 1.0], dtype=dtype)
    unit_y = jnp.array([0.95, 0.05, 0.51, 0.35, 1.0], dtype=dtype)
    tensor_samples = {
        name: jnp.array(values, dtype=dtype)
        for name, values in samples.items()
    }
    prior = distributions.FixedSamples(("x", "y"), tensor_samples)

    inverse = prior.cdfinv(x=unit_x, y=unit_y)
    draws = prior.rvs(size=32)

    for result in (inverse, draws):
        assert all(
            isinstance(value, (jnp.ndarray, jax.Array))
            for value in result.values()
        )
        assert all(value.dtype == dtype for value in result.values())
    paired = (draws["x"][:, None] == tensor_samples["x"][None, :]) & (
        draws["y"][:, None] == tensor_samples["y"][None, :]
    )
    assert bool(jnp.all(jnp.any(paired, axis=1)))


def test_convex_hull_constraint_preserves_boolean_tensor():
    hull_points = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    values = {
        "coeff_0": numpy.array([0.1, 0.9, 1.1, -0.1, numpy.nan]),
        "coeff_1": numpy.array([0.1, 0.9, 0.6, 0.2, 0.1]),
        "coeff_2": numpy.array([0.1, 0.9, 0.1, 0.2, 0.1]),
    }
    constraint = object.__new__(distributions.constraints.SupernovaeConvexHull)
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
    expected = numpy.array([mass_prior.logpdf(q=value) for value in q_values])
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


def _exact(actual, expected):
    actual, expected = numpy.asarray(actual), numpy.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
@pytest.mark.parametrize("operation", ["pdf", "logpdf", "contains"])
def test_original_scalar_prior_replay_is_exact(dtype, operation):
    from pycbc.scheme import JAXScheme

    prior = distributions.Gaussian(x=(-2.0, 2.0), x_mean=0.25, x_var=1.5)
    values = numpy.array([[-3.0, -0.2], [0.7, 2.5]], dtype=dtype)
    method = (
        (lambda **row: prior.__contains__(row))
        if operation == "contains"
        else getattr(prior, operation)
    )
    expected = numpy.array([method(x=x) for x in values.ravel()]).reshape(
        values.shape
    )
    with JAXScheme(
        "cpu", reference_operations=("priors.Gaussian." + operation,)
    ):
        result = method(x=jnp.asarray(values))
    _exact(result, expected)
    # Original NumPy vector behavior is unchanged by the new JAX batch API.
    with pytest.raises(ValueError, match="truth value"):
        method(x=values)


def test_original_gaussian_special_functions_replay_independently():
    from pycbc.scheme import JAXScheme

    prior = distributions.Gaussian(x=(-2.0, 2.0), x_mean=0.25, x_var=1.5)
    values = numpy.array([-1.2, -0.1, 0.7, 1.4])
    probabilities = numpy.array([0.1, 0.3, 0.5, 0.9])
    for name, value in (
        ("_normalcdf", values),
        ("_normalcdfinv", probabilities),
        ("cdf", values),
    ):
        function = getattr(prior, name)
        expected = function("x", value)
        with JAXScheme(
            "cpu", reference_operations=("priors.Gaussian." + name,)
        ):
            actual = function("x", jnp.asarray(value))
        _exact(actual, expected)


def test_one_prior_family_can_replay_while_another_stays_jax(monkeypatch):
    from pycbc.scheme import JAXScheme

    gaussian = distributions.Gaussian(x=(-2.0, 2.0), x_mean=0.25, x_var=1.5)
    power = distributions.UniformPowerLaw(dim=3, r=(1.0, 4.0))
    joint = distributions.JointDistribution(("x", "r"), gaussian, power)
    x = numpy.array([[-0.4], [0.7]])
    r = numpy.array([[1.2, 2.2, 3.2]])
    seen = []
    original = power._logpdf

    def record(**params):
        seen.append(isinstance(params["r"], jax.Array))
        return original(**params)

    monkeypatch.setattr(power, "_logpdf", record)
    with JAXScheme("cpu", reference_operations=("priors.Gaussian.logpdf",)):
        actual = joint(x=jnp.asarray(x), r=jnp.asarray(r))
        expected = (
            jnp.asarray(
                numpy.array([gaussian.logpdf(x=v) for v in x[:, 0]])[:, None]
            )
            + power.logpdf(r=jnp.asarray(r))
            - joint._logpdf_scale
        )
    assert seen and all(seen)
    _exact(actual, expected)


def test_original_mass_and_fixed_sample_inverse_cdfs_replay_exactly():
    from pycbc.scheme import JAXScheme

    mass = distributions.QfromUniformMass1Mass2(q=(1.0, 8.0))
    units = numpy.array([0.1, 0.4, 0.7, 0.9])
    expected_mass = mass.cdfinv(q=units)["q"]
    samples = {
        "x": numpy.array([1.0, 2.0, 3.0, 4.0]),
        "y": numpy.array([4.0, 3.0, 2.0, 1.0]),
    }
    fixed = distributions.FixedSamples(("x", "y"), samples)
    expected_fixed = [
        fixed.cdfinv(x=x, y=y) for x, y in zip(units, units[::-1])
    ]
    with JAXScheme(
        "cpu",
        reference_operations=(
            "priors.QfromUniformMass1Mass2.cdfinv",
            "priors.FixedSamples.cdfinv",
        ),
    ):
        _exact(mass.cdfinv(q=jnp.asarray(units))["q"], expected_mass)
        actual = fixed.cdfinv(x=jnp.asarray(units), y=jnp.asarray(units[::-1]))
    for name in samples:
        _exact(
            actual[name], numpy.array([row[name] for row in expected_fixed])
        )


def test_fixed_sample_replay_keeps_empty_batch_columns():
    from pycbc.scheme import JAXScheme

    samples = {"x": numpy.arange(1.0, 5.0), "y": numpy.arange(4.0, 0.0, -1.0)}
    prior = distributions.FixedSamples(("x", "y"), samples)
    with JAXScheme(
        "cpu", reference_operations=("priors.FixedSamples.cdfinv",)
    ):
        result = prior.cdfinv(x=jnp.empty((0, 2)), y=jnp.empty((0, 2)))
    assert isinstance(result, dict) and set(result) == set(samples)
    for name in samples:
        _exact(result[name], numpy.empty((0, 2), dtype=samples[name].dtype))


def test_gaussian_outside_bounds_keeps_original_short_circuit():
    prior = distributions.Gaussian(x=(-1.0, 1.0))
    with numpy.errstate(over="raise", invalid="raise"):
        assert prior.logpdf(x=numpy.float64(1e300)) == -numpy.inf


def test_native_record_membership_preserves_original_input_types():
    from pycbc.io.record import FieldArray

    prior = distributions.Gaussian(x=(-2.0, 2.0))
    field_array = FieldArray.from_kwargs(x=numpy.array([0.2]))
    record = numpy.array((0.2,), dtype=[("x", float)])[()]
    assert prior.__contains__(field_array) is True
    assert prior.__contains__(record) is True


def test_mass_constructor_keeps_native_spline_lazy(monkeypatch):
    from pycbc.distributions import mass

    def forbidden(*args, **kwargs):
        raise AssertionError("native construction invoked the JAX spline")

    monkeypatch.setattr(mass, "CubicSpline", forbidden)
    prior = distributions.QfromUniformMass1Mass2(q=(1.0, 8.0))
    assert numpy.isfinite(prior.cdfinv(q=0.4)["q"])


def test_kde_large_common_offset_and_first_jit_are_stable():
    rng = numpy.random.default_rng(123)
    samples = 1e8 + rng.normal(scale=0.01, size=64)
    prior = distributions.Arbitrary(x=samples)
    values = numpy.array([1e8 - 0.03, 1e8, 1e8 + 0.03])
    expected = numpy.log(prior.kde.evaluate(values))
    actual = prior.logpdf(x=jnp.asarray(values))
    # Different whitening order still rounds differently; cancellation must
    # not replace the entire density curve by its peak normalization.
    numpy.testing.assert_allclose(actual, expected, rtol=1e-5, atol=4e-6)
    fresh = distributions.Arbitrary(x=rng.normal(size=64))
    with jax.checking_leaks():
        compiled = jax.jit(lambda x: fresh.logpdf(x=x))(
            jnp.asarray([0.0, 1.0])
        )
    eager = fresh.logpdf(x=jnp.asarray([0.0, 1.0]))
    numpy.testing.assert_allclose(compiled, eager, rtol=1e-13, atol=1e-13)


def test_original_kde_scalar_exception_is_preserved():
    from pycbc.scheme import JAXScheme

    prior = distributions.Arbitrary(
        x=numpy.random.default_rng(3).normal(size=32)
    )
    try:
        expected = prior.logpdf(x=numpy.float64(0.2))
    except TypeError as exc:
        # Original PyCBC's float(length-one-array) cast fails with NumPy2.4+.
        with JAXScheme(
            "cpu", reference_operations=("priors.Arbitrary.logpdf",)
        ):
            with pytest.raises(TypeError, match=str(exc)):
                prior.logpdf(x=jnp.asarray(0.2))
    else:
        with JAXScheme(
            "cpu", reference_operations=("priors.Arbitrary.logpdf",)
        ):
            _exact(prior.logpdf(x=jnp.asarray(0.2)), expected)


def test_native_prior_replay_rejects_tracing_explicitly():
    from pycbc.scheme import JAXScheme

    prior = distributions.Gaussian(x=(-2.0, 2.0))
    with JAXScheme("cpu", reference_operations=("priors.Gaussian.logpdf",)):
        with pytest.raises(TypeError, match="requires concrete"):
            jax.grad(lambda x: prior.logpdf(x=x))(jnp.asarray(0.2))


def test_prior_constants_follow_raw_input_device_and_cache_separately():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two CPU devices")
    from pycbc.scheme import JAXScheme

    prior = distributions.Arbitrary(
        x=numpy.random.default_rng(4).normal(size=32)
    )
    for device in devices[:2]:
        value = jax.device_put(numpy.array([0.2, 0.7]), device)
        with JAXScheme("cpu:0"):
            result = prior.logpdf(x=value)
        assert result.device == device
    assert len(prior._jax_kde_cache) == 2


def test_ignored_jax_keyword_does_not_change_native_distribution():
    prior = distributions.Gaussian(x=(-2.0, 2.0))
    expected = prior.logpdf(x=0.2)
    actual = prior.logpdf(x=0.2, ignored=jnp.array([1.0, 2.0]))
    assert not isinstance(actual, jax.Array)
    _exact(actual, expected)
    contained = prior.__contains__(
        {"x": 0.2, "ignored": jnp.array([1.0, 2.0])}
    )
    assert contained is True


def test_kde_density_and_coordinate_controls_compose_independently(
    monkeypatch,
):
    from pycbc.domain_jax import native_result
    from pycbc.scheme import JAXScheme

    samples = numpy.linspace(0.05, 0.95, 48)
    prior = distributions.Arbitrary(bounds={"x": (0.0, 1.0)}, x=samples)
    values = numpy.array([0.15, 0.4, 0.8])
    transform = prior._transforms[prior._tparams["x"]]
    mapped = transform.logit(values, 0.0, 1.0)
    expected_density = prior.kde.evaluate(mapped)
    expected_jacobian = numpy.array(
        [transform.jacobian({"x": value}) for value in values]
    )
    original = prior.kde.evaluate
    calls = []

    def record(points):
        assert isinstance(points, numpy.ndarray)
        result = original(points)
        calls.append((points.copy(), result.copy()))
        return result

    monkeypatch.setattr(prior.kde, "evaluate", record)
    default = prior.logpdf(x=jnp.asarray(values))
    assert not calls
    assert bool(jnp.all(jnp.isfinite(default)))
    controls = (
        "priors.Arbitrary.kde",
        "transforms.Logit.logit",
        "transforms.Logit.jacobian",
    )
    with JAXScheme("cpu", reference_operations=controls):
        actual = prior.logpdf(x=jnp.asarray(values))
        density = native_result(
            "priors", "Arbitrary.kde", prior.kde.evaluate, jnp.asarray(mapped)
        )
        expected = jnp.log(density) + jnp.log(jnp.asarray(expected_jacobian))
    _exact(density, expected_density)
    _exact(actual, expected)
    assert len(calls) == 2
    _exact(calls[0][0], mapped[None, :])
    _exact(calls[0][1], expected_density)


def test_bounded_kde_first_jit_does_not_cache_tracers():
    prior = distributions.Arbitrary(
        bounds={"x": (0.0, 1.0)}, x=numpy.linspace(0.05, 0.95, 48)
    )
    values = jnp.array([0.15, 0.4, 0.8])
    with jax.checking_leaks():
        compiled = jax.jit(lambda x: prior.logpdf(x=x))(values)
    eager = prior.logpdf(x=values)
    numpy.testing.assert_allclose(compiled, eager, rtol=1e-13, atol=1e-13)
    gradient = jax.grad(lambda x: jnp.sum(prior.logpdf(x=x)))(values)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_gaussian_mask_has_zero_gradient_outside_bounds():
    prior = distributions.Gaussian(x=(-1.0, 1.0))
    for value in (2.0, 1e300, numpy.nan):
        gradient = jax.grad(lambda x: prior.logpdf(x=x))(jnp.asarray(value))
        assert float(gradient) == 0.0
