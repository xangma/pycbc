"""Fixed CPython binary64 XAS intermediate-amplitude lane.

Keep this module free of Torch imports. The caller imports it only after the
independent, off-by-default gate and exact runtime contract have both passed.
The expression topology mirrors ``_prepare_intermediate_amp`` deliberately.

The caller supplies every fractional power and exponential after evaluating
them with Torch. Fractional powers use Torch's foreach Tensor-Scalar path so
one Python dispatch preserves each original scalar-power overload. Integer
powers use explicit multiplication because Torch's
scalar ``pow`` implements powers two and three that way. This function therefore
executes only ordered binary64 addition, subtraction, multiplication, division,
and negation in CPython.
"""


def intermediate_amp_lane(
    FMs1,
    FMs4,
    V2,
    A2,
    A3,
    A4,
    A5,
    A6,
    rho1,
    rho2,
    rho3,
    fMs_RD,
    gammaR,
    gammaD2,
    gammaD13,
    exact_powers,
    exact_exponential,
):
    A0 = 1.0
    inspFMs1 = (
        A0
        + A2 * exact_powers[0]
        + A3 * FMs1
        + A4 * exact_powers[1]
        + A5 * exact_powers[2]
        + A6 * (FMs1 * FMs1)
        + rho1 * exact_powers[3]
        + rho2 * exact_powers[4]
        + rho3 * ((FMs1 * FMs1) * FMs1)
    )
    c0 = A2 * ((2.0 / 3.0) * exact_powers[5])
    c1 = A3
    c2 = A4 * ((4.0 / 3.0) * exact_powers[6])
    c3 = A5 * ((5.0 / 3.0) * exact_powers[7])
    c4 = A6 * (2.0 * FMs1)
    c5 = rho1 * ((7.0 / 3.0) * exact_powers[8])
    c6 = rho2 * ((8.0 / 3.0) * exact_powers[9])
    c7 = rho3 * (3.0 * (FMs1 * FMs1))
    d1 = c7 + c6
    d1 = d1 + c5
    d1 = d1 + c4
    d1 = d1 + c3
    d1 = d1 + c2
    d1 = d1 + c1
    d1 = d1 + c0

    left_offset = FMs4 - fMs_RD
    right_offset = FMs4 - fMs_RD
    exponential = exact_exponential
    numerator = exponential * gammaD13
    denominator = left_offset * right_offset + gammaD2
    rdFMs4 = numerator / denominator

    one = 1.0
    numerator_derivative = -(
        (((one / denominator) * gammaD13) * exponential) * gammaR
    )
    denominator_adjoint = (-(numerator / denominator)) / denominator
    denominator_factor_derivative = denominator_adjoint * left_offset
    d4 = (
        denominator_factor_derivative + denominator_factor_derivative
    ) + numerator_derivative

    d1 = ((7.0 / 6.0) * exact_powers[10] / inspFMs1) - (
        exact_powers[11] * d1 / (inspFMs1 * inspFMs1)
    )
    d4 = ((7.0 / 6.0) * exact_powers[12] / rdFMs4) - (
        exact_powers[13] * d4 / (rdFMs4 * rdFMs4)
    )

    FMs2 = FMs1 + (1.0 / 2.0) * (FMs4 - FMs1)
    V1 = exact_powers[14] * inspFMs1
    V4 = exact_powers[15] * rdFMs4
    V1 = 1.0 / V1
    V2 = 1.0 / V2
    V4 = 1.0 / V4
    V2 = V2 + 0.0

    F12 = FMs1 * FMs1
    F13 = F12 * FMs1
    F14 = F13 * FMs1
    F15 = F14 * FMs1

    F22 = FMs2 * FMs2
    F23 = F22 * FMs2
    F24 = F23 * FMs2

    F42 = FMs4 * FMs4
    F43 = F42 * FMs4
    F44 = F43 * FMs4
    F45 = F44 * FMs4

    F1mF2 = FMs1 - FMs2
    F1mF4 = FMs1 - FMs4
    F2mF4 = FMs2 - FMs4

    F1mF22 = F1mF2 * F1mF2
    F2mF42 = F2mF4 * F2mF4
    F1mF43 = F1mF4 * F1mF4 * F1mF4

    delta0 = (
        -(d4 * F12 * F1mF22 * F1mF4 * FMs2 * F2mF4 * FMs4)
        + d1 * FMs1 * F1mF2 * F1mF4 * FMs2 * F2mF42 * F42
        + F42
        * (
            FMs2
            * F2mF42
            * (-4 * F12 + 3 * FMs1 * FMs2 + 2 * FMs1 * FMs4 - FMs2 * FMs4)
            * V1
            + F12 * F1mF43 * V2
        )
        + F12
        * F1mF22
        * FMs2
        * (FMs1 * FMs2 - 2 * FMs1 * FMs4 - 3 * FMs2 * FMs4 + 4 * F42)
        * V4
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta1 = (
        d4 * FMs1 * F1mF22 * F1mF4 * F2mF4 * (2 * FMs2 * FMs4 + FMs1 * (FMs2 + FMs4))
        + FMs4
        * (
            -(d1 * F1mF2 * F1mF4 * F2mF42 * (2 * FMs1 * FMs2 + (FMs1 + FMs2) * FMs4))
            - 2
            * FMs1
            * (
                F44 * (V1 - V2)
                + 3 * F24 * (V1 - V4)
                + F14 * (V2 - V4)
                + 4 * F23 * FMs4 * (-V1 + V4)
                + 2 * F13 * FMs4 * (-V2 + V4)
                + FMs1
                * (
                    2 * F43 * (-V1 + V2)
                    + 6 * F22 * FMs4 * (V1 - V4)
                    + 4 * F23 * (-V1 + V4)
                )
            )
        )
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta2 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4 * (F12 + FMs2 * FMs4 + 2 * FMs1 * (FMs2 + FMs4)))
        + d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1 * FMs2 + 2 * (FMs1 + FMs2) * FMs4 + F42)
        - 4 * F12 * F23 * V1
        + 3 * FMs1 * F24 * V1
        - 4 * FMs1 * F23 * FMs4 * V1
        + 3 * F24 * FMs4 * V1
        + 12 * F12 * FMs2 * F42 * V1
        - 4 * F23 * F42 * V1
        - 8 * F12 * F43 * V1
        + FMs1 * F44 * V1
        + F45 * V1
        + F15 * V2
        + F14 * FMs4 * V2
        - 8 * F13 * F42 * V2
        + 8 * F12 * F43 * V2
        - FMs1 * F44 * V2
        - F45 * V2
        - F1mF22
        * (
            F13
            + FMs2 * (3 * FMs2 - 4 * FMs4) * FMs4
            + F12 * (2 * FMs2 + FMs4)
            + FMs1 * (3 * FMs2 - 4 * FMs4) * (FMs2 + 2 * FMs4)
        )
        * V4
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta3 = (
        d4 * F1mF22 * F1mF4 * F2mF4 * (2 * FMs1 + FMs2 + FMs4)
        - d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1 + FMs2 + 2 * FMs4)
        + 2
        * (
            F44 * (-V1 + V2)
            + 2 * F12 * F2mF42 * (V1 - V4)
            + 2 * F22 * F42 * (V1 - V4)
            + 2 * F13 * FMs4 * (V2 - V4)
            + F24 * (-V1 + V4)
            + F14 * (-V2 + V4)
            + 2
            * FMs1
            * FMs4
            * (F42 * (V1 - V2) + F22 * (V1 - V4) + 2 * FMs2 * FMs4 * (-V1 + V4))
        )
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta4 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4)
        + d1 * F1mF2 * F1mF4 * F2mF42
        - 3 * FMs1 * F22 * V1
        + 2 * F23 * V1
        + 6 * FMs1 * FMs2 * FMs4 * V1
        - 3 * F22 * FMs4 * V1
        - 3 * FMs1 * F42 * V1
        + F43 * V1
        + F13 * V2
        - 3 * F12 * FMs4 * V2
        + 3 * FMs1 * F42 * V2
        - F43 * V2
        - F1mF22 * (FMs1 + 2 * FMs2 - 3 * FMs4) * V4
    ) / (F1mF22 * F1mF43 * F2mF42)

    return delta0, delta1, delta2, delta3, delta4
