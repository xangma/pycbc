# Copyright (C) 2026  PyCBC developers
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Focused Torch compatibility tests for inference prior distributions."""

import numpy
import pytest

from pycbc import distributions, transforms

torch = pytest.importorskip("torch")


@pytest.fixture(
    params=["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
)
def device(request):
    return torch.device(request.param)


def _assert_matches(result, expected, device, *, rtol=1e-11, atol=1e-12):
    assert isinstance(result, torch.Tensor)
    assert result.device == device
    numpy.testing.assert_allclose(
        result.detach().cpu().numpy(), expected, rtol=rtol, atol=atol,
        equal_nan=True
    )


def test_common_and_joint_priors_preserve_tensors(device):
    dtype = torch.float64
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

    x = torch.tensor(x_values, dtype=dtype, device=device, requires_grad=True)
    radius = torch.tensor(
        r_values, dtype=dtype, device=device, requires_grad=True
    )
    scale = torch.tensor(
        scale_values, dtype=dtype, device=device, requires_grad=True
    )
    gaussian_log = gaussian.logpdf(x=x)
    power_log = power.logpdf(r=radius)
    scale_log = log_uniform.logpdf(scale=scale)
    uniform_log = uniform.logpdf(
        u=torch.as_tensor(u_values, dtype=dtype, device=device),
        v=torch.as_tensor(v_values, dtype=dtype, device=device),
    )
    joint_log = joint(x=x, r=radius)
    contains = joint.contains({"x": x, "r": radius})
    constrained = joint.within_constraints({"x": x, "r": radius})

    _assert_matches(gaussian_log, expected_gaussian, device)
    _assert_matches(power_log, expected_power, device)
    _assert_matches(scale_log, expected_scale, device)
    _assert_matches(uniform_log, expected_uniform, device)
    assert contains.dtype == torch.bool
    assert contains.tolist() == [False, True, False]
    assert constrained.dtype == torch.bool
    assert constrained.tolist() == [False, True, False]
    assert torch.equal(contains, constrained)
    assert torch.equal(torch.isfinite(joint_log), constrained)

    (gaussian_log.sum() + power_log.sum() + scale_log.sum()).backward()
    torch.testing.assert_close(x.grad, -(x.detach() - 0.25) / 1.5)
    torch.testing.assert_close(radius.grad, 2.0 / radius.detach())
    torch.testing.assert_close(scale.grad, -1.0 / scale.detach())


def test_transformed_and_angular_constraints_preserve_boolean_tensors(device):
    dtype = torch.float64
    shifted = transforms.CustomTransform(
        ["x"], ["shifted_x"], {"shifted_x": "x + 0.5"}
    )
    transformed = distributions.constraints.Constraint(
        "shifted_x < 1.0", transforms=[shifted]
    )
    angular = distributions.constraints.Constraint(
        "(dec + ddec >= -pi/2) & (dec + ddec <= pi/2)"
    )
    x = torch.tensor([-1.5, -0.25, 0.75], dtype=dtype, device=device)
    dec = torch.tensor([-1.4, 0.0, 1.4], dtype=dtype, device=device)
    ddec = torch.tensor([-0.3, 0.2, 0.3], dtype=dtype, device=device)

    transformed_result = transformed({"x": x})
    angular_result = angular({"dec": dec, "ddec": ddec})

    assert transformed_result.device == device
    assert transformed_result.dtype == torch.bool
    assert transformed_result.tolist() == [True, True, False]
    assert angular_result.device == device
    assert angular_result.dtype == torch.bool
    assert angular_result.tolist() == [False, True, False]


def test_angular_priors_and_inverse_cdfs_preserve_gradients(device):
    dtype = torch.float64
    sin_prior = distributions.SinAngle(theta=(0.2, 2.7))
    cos_prior = distributions.CosAngle(declination=(-1.2, 1.1))
    solid_prior = distributions.UniformSolidAngle(
        polar_bounds=(0.1, 0.8), azimuthal_bounds=(0.2, 1.7)
    )
    theta_values = numpy.array([0.4, 1.2, 2.6])
    declination_values = numpy.array([-1.1, 0.1, 1.0])
    unit_values = numpy.array([0.05, 0.27, 0.61])
    azimuthal_values = numpy.array([0.91, 0.48, 0.13])
    expected_sin = numpy.array(
        [sin_prior.logpdf(theta=value) for value in theta_values]
    )
    expected_cos = numpy.array(
        [cos_prior.logpdf(declination=value)
         for value in declination_values]
    )
    expected_solid = solid_prior.cdfinv(
        theta=unit_values, phi=azimuthal_values
    )

    theta = torch.tensor(
        theta_values, dtype=dtype, device=device, requires_grad=True
    )
    declination = torch.tensor(
        declination_values, dtype=dtype, device=device, requires_grad=True
    )
    unit = torch.tensor(
        unit_values, dtype=dtype, device=device, requires_grad=True
    )
    azimuthal = torch.tensor(
        azimuthal_values, dtype=dtype, device=device, requires_grad=True
    )
    sin_log = sin_prior.logpdf(theta=theta)
    cos_log = cos_prior.logpdf(declination=declination)
    solid = solid_prior.cdfinv(theta=unit, phi=azimuthal)

    _assert_matches(sin_log, expected_sin, device)
    _assert_matches(cos_log, expected_cos, device)
    _assert_matches(solid["theta"], expected_solid["theta"], device)
    _assert_matches(solid["phi"], expected_solid["phi"], device)
    (sin_log.sum() + cos_log.sum() + solid["theta"].sum()
     + solid["phi"].sum()).backward()
    assert all(value.grad is not None
               for value in (theta, declination, unit, azimuthal))


def test_kde_and_spin_priors_preserve_tensors(device):
    dtype = torch.float64
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
    x = torch.tensor(
        x_values, dtype=dtype, device=device, requires_grad=True
    )
    y = torch.as_tensor(y_values, dtype=dtype, device=device)

    actual = prior.logpdf(x=x, y=y)

    _assert_matches(actual, expected, device, rtol=1e-10)
    actual.sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())
    assert prior._torch_kde_cache
    prior.set_bandwidth("silverman")
    assert not prior._torch_kde_cache

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
        name: torch.tensor(values, dtype=dtype, device=device)
        for name, values in spin_values.items()
    }
    spin_contains = spin_prior.__contains__(spin_tensors)
    assert spin_contains.device == device
    assert spin_contains.dtype == torch.bool
    assert spin_contains.tolist() == [True, False]


def test_tabulated_prior_interpolation_preserves_gradients(tmp_path, device):
    dtype = torch.float64
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
    value_tensor = torch.tensor(
        values, dtype=dtype, device=device, requires_grad=True
    )
    unit_tensor = torch.tensor(
        units, dtype=dtype, device=device, requires_grad=True
    )

    actual_pdf = prior._pdf(value_tensor)
    actual_cdf = prior._cdf(value_tensor)
    actual_inverse = prior.cdfinv(x=unit_tensor)["x"]

    _assert_matches(actual_pdf, expected_pdf, device)
    _assert_matches(actual_cdf, expected_cdf, device)
    _assert_matches(actual_inverse, expected_inverse, device)
    (actual_pdf.sum() + actual_cdf.sum() + actual_inverse.sum()).backward()
    assert value_tensor.grad is not None
    assert unit_tensor.grad is not None
    assert bool(torch.isfinite(value_tensor.grad).all())
    assert bool(torch.isfinite(unit_tensor.grad).all())


def test_fixed_samples_preserve_pairing_and_device(device):
    dtype = torch.float64
    samples = {
        "x": numpy.array([-4.0, -2.0, -1.0, 0.5, 1.5, 3.0]),
        "y": numpy.array([9.0, -3.0, 8.0, 1.0, -4.0, 6.0]),
    }
    unit_x = torch.tensor(
        [0.0, 0.18, 0.45, 0.72, 1.0], dtype=dtype, device=device
    )
    unit_y = torch.tensor(
        [0.95, 0.05, 0.51, 0.35, 1.0], dtype=dtype, device=device
    )
    tensor_samples = {
        name: torch.as_tensor(values, dtype=dtype, device=device)
        for name, values in samples.items()
    }
    prior = distributions.FixedSamples(("x", "y"), tensor_samples)

    inverse = prior.cdfinv(x=unit_x, y=unit_y)
    draws = prior.rvs(size=32)

    for result in (inverse, draws):
        assert all(value.device == device for value in result.values())
        assert all(value.dtype == dtype for value in result.values())
    paired = (
        (draws["x"][:, None] == tensor_samples["x"][None, :])
        & (draws["y"][:, None] == tensor_samples["y"][None, :])
    )
    assert bool(paired.any(dim=1).all())


def test_convex_hull_constraint_preserves_boolean_tensor(device):
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
    constraint._torch_hull_cache = {}
    constraint._torch_max_working_elements = 3
    expected = constraint(values)
    tensor_values = {
        name: torch.as_tensor(value, dtype=torch.float64, device=device)
        for name, value in values.items()
    }

    actual = constraint(tensor_values)

    assert actual.device == device
    assert actual.dtype == torch.bool
    assert actual.detach().cpu().tolist() == expected.tolist()
    assert len(constraint._torch_hull_cache) == 1


def test_mass_ratio_and_qnm_priors_accept_tensor_batches(device):
    dtype = torch.float64
    mass_prior = distributions.QfromUniformMass1Mass2(q=(1.0, 8.0))
    q_values = numpy.array([1.0, 2.5, 7.5])
    expected = numpy.array(
        [mass_prior.logpdf(q=value) for value in q_values]
    )
    q = torch.tensor(
        q_values, dtype=dtype, device=device, requires_grad=True
    )

    logpdf = mass_prior.logpdf(q=q)

    _assert_matches(logpdf, expected, device)
    logpdf.sum().backward()
    assert q.grad is not None
    assert bool(torch.isfinite(q.grad).all())

    qnm_prior = distributions.UniformF0Tau(
        f0=(100.0, 500.0), tau=(0.001, 0.02), norm_tolerance=0.1
    )
    f0 = torch.tensor([200.0, 80.0], dtype=dtype, device=device)
    tau = torch.tensor([0.004, 0.004], dtype=dtype, device=device)
    contained = qnm_prior.__contains__({"f0": f0, "tau": tau})
    assert contained.device == device
    assert contained.dtype == torch.bool
    assert contained.tolist() == [True, False]
