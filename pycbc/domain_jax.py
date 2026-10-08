# Copyright (C) 2026 PyCBC developers
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional original-kernel validation for numerical domain APIs.

Importing the operation names or decorators does not import JAX. Native
validation deliberately uses writable host copies and is not differentiable.
"""

from functools import wraps

DOMAIN_REFERENCE_NAMES = {
    "priors": tuple(
        f"{family}.{operation}"
        for family in (
            "BoundedDist", "Uniform", "UniformAngle", "SinAngle", "CosAngle",
            "UniformSolidAngle", "Gaussian", "UniformPowerLaw", "UniformRadius",
            "UniformLog10", "MchirpfromUniformMass1Mass2", "QfromUniformMass1Mass2",
            "UniformF0Tau", "Arbitrary", "FromFile", "IndependentChiPChiEff",
        )
        for operation in ("pdf", "logpdf", "cdfinv", "contains")
    ) + (
        "Gaussian.cdf", "Gaussian._normalcdf", "Gaussian._normalcdfinv",
        "JointDistribution.logpdf", "JointDistribution.contains",
        "JointDistribution.within_constraints", "Constraint.evaluate",
        "SupernovaeConvexHull.evaluate", "FixedSamples.cdfinv",
        "DistributionFunctionFromFile.pdf", "DistributionFunctionFromFile.logpdf",
        "DistributionFunctionFromFile.cdf", "DistributionFunctionFromFile.cdfinv",
        "Arbitrary.kde",
    ),
    "conversions": (
        "sec_to_year",
        "hypertriangle",
        "primary_mass",
        "secondary_mass",
        "mtotal_from_mass1_mass2",
        "q_from_mass1_mass2",
        "invq_from_mass1_mass2",
        "eta_from_mass1_mass2",
        "mchirp_from_mass1_mass2",
        "eccmchirp_from_mass1_mass2_eccentricity",
        "mass1_from_mtotal_q",
        "mass2_from_mtotal_q",
        "mass1_from_mtotal_eta",
        "mass2_from_mtotal_eta",
        "mtotal_from_mchirp_eta",
        "mass1_from_mchirp_eta",
        "mass2_from_mchirp_eta",
        "mass2_from_mchirp_mass1",
        "mass_from_knownmass_eta",
        "mass2_from_mass1_eta",
        "mass1_from_mass2_eta",
        "eta_from_q",
        "mass1_from_mchirp_q",
        "mass2_from_mchirp_q",
        "tau0_from_mtotal_eta",
        "tau0_from_mchirp",
        "tau3_from_mtotal_eta",
        "tau0_from_mass1_mass2",
        "tau3_from_mass1_mass2",
        "mchirp_from_tau0",
        "mtotal_from_tau0_tau3",
        "eta_from_tau0_tau3",
        "mass1_from_tau0_tau3",
        "mass2_from_tau0_tau3",
        "eccmchirp_from_mchirp_eccentricity",
        "lambda_tilde",
        "delta_lambda_tilde",
        "lambda1_from_delta_lambda_tilde_lambda_tilde",
        "lambda2_from_delta_lambda_tilde_lambda_tilde",
        "lambda_from_mass_tov_file",
        "ensure_obj1_is_primary",
        "remnant_mass_from_mass1_mass2_spherical_spin_eos",
        "remnant_mass_from_mass1_mass2_cartesian_spin_eos",
        "chi_eff",
        "chi_a",
        "chi_p",
        "phi_a",
        "phi_s",
        "chi_eff_from_spherical",
        "chi_p_from_spherical",
        "primary_spin",
        "secondary_spin",
        "primary_xi",
        "secondary_xi",
        "xi1_from_spin1x_spin1y",
        "xi2_from_mass1_mass2_spin2x_spin2y",
        "chi_perp_from_spinx_spiny",
        "chi_perp_from_mass1_mass2_xi2",
        "chi_p_from_xi1_xi2",
        "phi1_from_phi_a_phi_s",
        "phi2_from_phi_a_phi_s",
        "phi_from_spinx_spiny",
        "spin1z_from_mass1_mass2_chi_eff_chi_a",
        "spin2z_from_mass1_mass2_chi_eff_chi_a",
        "spin1x_from_xi1_phi_a_phi_s",
        "spin1y_from_xi1_phi_a_phi_s",
        "spin2x_from_mass1_mass2_xi2_phi_a_phi_s",
        "spin2y_from_mass1_mass2_xi2_phi_a_phi_s",
        "dquadmon_from_lambda",
        "spin_from_pulsar_freq",
        "chirp_distance",
        "distance_from_chirp_distance_mchirp",
        "snr_from_loglr",
        "get_lm_f0tau",
        "get_lm_f0tau_allmodes",
        "freq_from_final_mass_spin",
        "tau_from_final_mass_spin",
        "final_spin_from_f0_tau",
        "final_mass_from_f0_tau",
        "freqlmn_from_other_lmn",
        "taulmn_from_other_lmn",
        "velocity_to_frequency",
        "frequency_to_velocity",
        "f_schwarzchild_isco",
        "nltides_coefs",
        "nltides_gw_phase_difference",
        "nltides_gw_phase_diff_isco",
    ),
    "coordinates": (
        "cartesian_to_spherical_rho",
        "cartesian_to_spherical_azimuthal",
        "cartesian_to_spherical_polar",
        "cartesian_to_spherical",
        "spherical_to_cartesian",
    ),
    "cosmology": (
        "redshift",
        "redshift_from_comoving_volume",
        "distance_from_comoving_volume",
        "DistToZ.get_redshift",
        "ComovingVolInterpolator.get_value_from_logv",
        "ComovingVolInterpolator.get_value",
    ),
    "transforms": (
        "CustomTransform.transform",
        "CustomTransform.jacobian",
        "CustomTransformMultiOutputs.transform",
        "MchirpQToMass1Mass2.transform",
        "MchirpQToMass1Mass2.inverse_transform",
        "MchirpQToMass1Mass2.jacobian",
        "MchirpQToMass1Mass2.inverse_jacobian",
        "MchirpEtaToMass1Mass2.transform",
        "MchirpEtaToMass1Mass2.inverse_transform",
        "MchirpEtaToMass1Mass2.jacobian",
        "MchirpEtaToMass1Mass2.inverse_jacobian",
        "ChirpDistanceToDistance.transform",
        "ChirpDistanceToDistance.inverse_transform",
        "ChirpDistanceToDistance.jacobian",
        "ChirpDistanceToDistance.inverse_jacobian",
        "SphericalToCartesian.transform",
        "SphericalToCartesian.inverse_transform",
        "DistanceToRedshift.transform",
        "AlignedMassSpinToCartesianSpin.transform",
        "AlignedMassSpinToCartesianSpin.inverse_transform",
        "PrecessionMassSpinToCartesianSpin.transform",
        "PrecessionMassSpinToCartesianSpin.inverse_transform",
        "CartesianSpinToChiP.transform",
        "LambdaFromTOVFile.transform",
        "LambdaFromMultipleTOVFiles.transform",
        "Log.transform",
        "Log.inverse_transform",
        "Log.jacobian",
        "Log.inverse_jacobian",
        "Logit.logit",
        "Logit.logistic",
        "Logit.transform",
        "Logit.inverse_transform",
        "Logit.jacobian",
        "Logit.inverse_jacobian",
    ),
    "boundaries": (
        "apply_cyclic",
        "Bounds.apply_conditions",
        "Bounds.contains_conditioned",
    ),
}

DOMAIN_REFERENCE_OPERATIONS = frozenset(DOMAIN_REFERENCE_NAMES) | frozenset(
    f"{category}.{name}"
    for category, names in DOMAIN_REFERENCE_NAMES.items()
    for name in names
)

REFERENCE_UNSELECTED = object()


def _jax_reference(value):
    from pycbc.types.backend import backend_array, jax_module_for

    storage = backend_array(value)
    if jax_module_for(storage) is not None:
        return storage
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        return None
    for item in values:
        reference = _jax_reference(item)
        if reference is not None:
            return reference
    return None


def _host_copy(value, aliases):
    import numpy
    from pycbc.types.backend import backend_array, jax_module_for

    storage = backend_array(value)
    if jax_module_for(storage) is not None or isinstance(value, numpy.ndarray):
        copied = (value.copy(order="K") if isinstance(value, numpy.ndarray)
                  else numpy.array(storage, copy=True, order="K"))
        aliases[id(copied)] = value
        return copied
    if isinstance(value, dict):
        return {key: _host_copy(item, aliases) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_host_copy(item, aliases) for item in value)
    if isinstance(value, list):
        return [_host_copy(item, aliases) for item in value]
    return value


def _device_result(value, reference, aliases):
    import numpy
    import jax

    if id(value) in aliases:
        return aliases[id(value)]
    if isinstance(value, dict):
        return {key: _device_result(item, reference, aliases)
                for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_device_result(item, reference, aliases) for item in value)
    if isinstance(value, list):
        return [_device_result(item, reference, aliases) for item in value]
    if isinstance(value, (numpy.ndarray, numpy.number, float, int, complex, bool)):
        # Do not cast to the JAX input dtype: preserve the original result's
        # dtype and bytes, including NumPy/SciPy's higher-precision outputs.
        array = numpy.asarray(value)
        if array.dtype.kind not in "biufc":
            return value
        if not jax.config.x64_enabled and (
            (array.dtype.kind in "iuf" and array.dtype.itemsize > 4)
            or (array.dtype.kind == "c" and array.dtype.itemsize > 8)
        ):
            raise RuntimeError(
                "Original domain validation requires jax_enable_x64=True "
                f"to preserve the native {array.dtype} result"
            )
        return jax.device_put(array, reference.sharding)
    return value


def native_result(category, name, function, *args, host_function=None, **kwargs):
    """Call ``function`` with native inputs if its reference control is selected.

    Return ``REFERENCE_UNSELECTED`` when the default path should run. This
    never changes the scheme or catches/reinterprets the original exception.
    """
    from pycbc import scheme

    selected = getattr(scheme.mgr.state, "jax_reference_operations", ())
    if category not in selected and f"{category}.{name}" not in selected:
        return REFERENCE_UNSELECTED
    reference = _jax_reference((args, kwargs))
    if reference is None:
        return REFERENCE_UNSELECTED
    import jax

    if isinstance(reference, jax.core.Tracer):
        raise TypeError("Original domain validation requires concrete JAX inputs")
    aliases = {}
    host_args, host_kwargs = _host_copy(args, aliases), _host_copy(kwargs, aliases)
    if host_function is None:
        result = function(*host_args, **host_kwargs)
    else:
        result = host_function(function, *host_args, **host_kwargs)
    return _device_result(result, reference, aliases)


def reference(category, name, host_function=None):
    """Decorate one original public API with an independent validation route."""
    def decorate(function):
        @wraps(function)
        def dispatch(*args, **kwargs):
            result = native_result(category, name, function, *args,
                                   host_function=host_function, **kwargs)
            if result is REFERENCE_UNSELECTED:
                return function(*args, **kwargs)
            return result
        return dispatch
    return decorate
