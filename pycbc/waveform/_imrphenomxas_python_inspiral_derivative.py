"""Ordered binary64 executor for one scalar XAS inspiral reverse pass.

This module deliberately contains no Torch operations.  Its caller supplies
the values produced by Torch's nonlinear primitives, then qualifies and
materializes the result.  Keeping every statement in the eager forward and
reverse order is required for raw-byte parity at higher-mode matching seams.
"""


def inspiral_phase_value_and_derivative_lane(
    frequency,
    plan,
    output_adjoint,
    initial_gradient,
    f13,
    fminus23,
    log_f,
    phase_normalization,
):
    """Return the ordered scalar phase value and frequency derivative."""

    (
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
        phi5_l,
        phi6,
        phi6_l,
        phi7,
        phi8,
        phi8_l,
        sigma1,
        sigma2,
        sigma3,
        sigma4,
    ) = plan

    f23 = f13 * f13
    f43 = frequency * f13
    f53 = frequency * f23
    f2 = frequency * frequency
    f73 = f2 * f13
    f83 = f2 * f23
    f3 = f2 * frequency
    f103 = f3 * f13
    f113 = f3 * f23

    term1 = phi1 * f13
    phase_tf2 = phi0 + term1
    term2 = phi2 * f23
    phase_tf2 = phase_tf2 + term2
    term3 = phi3 * frequency
    phase_tf2 = phase_tf2 + term3
    term4 = phi4 * f43
    phase_tf2 = phase_tf2 + term4
    term5 = phi5 * f53
    phase_tf2 = phase_tf2 + term5
    term5_l_pre = phi5_l * f53
    term5_l = term5_l_pre * log_f
    phase_tf2 = phase_tf2 + term5_l
    term6 = phi6 * f2
    phase_tf2 = phase_tf2 + term6
    term6_l_pre = phi6_l * f2
    term6_l = term6_l_pre * log_f
    phase_tf2 = phase_tf2 + term6_l
    term7 = phi7 * f73
    phase_tf2 = phase_tf2 + term7
    term8 = phi8 * f83
    phase_tf2 = phase_tf2 + term8
    term8_l_pre = phi8_l * f83
    term8_l = term8_l_pre * log_f
    phase_tf2 = phase_tf2 + term8_l

    sigma_term1 = sigma1 * f83
    sigma_term2 = sigma2 * f3
    sigma_phase = sigma_term1 + sigma_term2
    sigma_term3 = sigma3 * f103
    sigma_phase = sigma_phase + sigma_term3
    sigma_term4 = sigma4 * f113
    sigma_phase = sigma_phase + sigma_term4
    phase_inspiral = phase_tf2 + sigma_phase
    scaled_phase = phase_inspiral * phase_normalization
    value = scaled_phase / f53

    one = 1.0 if output_adjoint is None else output_adjoint
    gradient_scaled_phase = one / f53
    gradient_f53 = -one * ((scaled_phase / f53) / f53)
    gradient_phase = gradient_scaled_phase * phase_normalization

    gradient_f113 = gradient_phase * sigma4
    gradient_f103 = gradient_phase * sigma3
    gradient_f3 = gradient_phase * sigma2
    gradient_f83 = gradient_phase * sigma1

    gradient_pre = gradient_phase * log_f
    gradient_log = gradient_phase * term8_l_pre
    gradient_f83 = gradient_f83 + gradient_pre * phi8_l
    gradient_f83 = gradient_f83 + gradient_phase * phi8
    gradient_f73 = gradient_phase * phi7
    gradient_pre = gradient_phase * log_f
    gradient_log = gradient_log + gradient_phase * term6_l_pre
    gradient_f2 = gradient_pre * phi6_l
    gradient_f2 = gradient_f2 + gradient_phase * phi6
    gradient_pre = gradient_phase * log_f
    gradient_log = gradient_log + gradient_phase * term5_l_pre
    gradient_f53 = gradient_f53 + gradient_pre * phi5_l
    gradient_f53 = gradient_f53 + gradient_phase * phi5
    gradient_f43 = gradient_phase * phi4
    direct_gradient = gradient_phase * phi3
    gradient_frequency = (
        direct_gradient
        if initial_gradient is None
        else initial_gradient + direct_gradient
    )
    gradient_f23 = gradient_phase * phi2
    gradient_f13 = gradient_phase * phi1

    gradient_frequency = gradient_frequency + gradient_log / frequency
    gradient_f3 = gradient_f3 + gradient_f113 * f23
    gradient_f23 = gradient_f23 + gradient_f113 * f3
    gradient_f3 = gradient_f3 + gradient_f103 * f13
    gradient_f13 = gradient_f13 + gradient_f103 * f3
    gradient_f2 = gradient_f2 + gradient_f3 * frequency
    gradient_frequency = gradient_frequency + gradient_f3 * f2
    gradient_f2 = gradient_f2 + gradient_f83 * f23
    gradient_f23 = gradient_f23 + gradient_f83 * f2
    gradient_f2 = gradient_f2 + gradient_f73 * f13
    gradient_f13 = gradient_f13 + gradient_f73 * f2
    gradient_frequency = gradient_frequency + gradient_f2 * frequency
    gradient_frequency = gradient_frequency + gradient_f2 * frequency
    gradient_frequency = gradient_frequency + gradient_f53 * f23
    gradient_f23 = gradient_f23 + gradient_f53 * frequency
    gradient_frequency = gradient_frequency + gradient_f43 * f13
    gradient_f13 = gradient_f13 + gradient_f43 * frequency
    gradient_f13 = gradient_f13 + gradient_f23 * f13
    gradient_f13 = gradient_f13 + gradient_f23 * f13
    gradient_frequency = gradient_frequency + gradient_f13 * (
        (1.0 / 3.0) * fminus23
    )
    return value, gradient_frequency
