# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Torch kernels for analytical ground-detector PSD models."""

import importlib.util
import math
import os

import lal
import numpy
import torch

from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


# Advanced-LIGO constants used by the analytical quantum and thermal models.
_ARM_LENGTH = 3995.0
_LASER_WAVELENGTH = 1.064e-6
_MIRROR_MASS = 40.0
_MIRROR_LOSS = 37.5e-6
_BEAM_SPLITTER_LOSS = 0.002
_ITM_TRANSMITTANCE = 0.014
_PRM_TRANSMITTANCE = 0.027
_TEMPERATURE = 290.0
_SUSPENSION_FREQUENCY = 9.0
_SUSPENSION_QUALITY = 6e10
_COATING_FREQUENCY = 1e4
_COATING_QUALITY = 6e6

# Initial- and enhanced-LIGO constants from LALSimNoisePSD.c.
_ILIGO_ARM_LENGTH = 3995.0
_ILIGO_LASER_POWER_BS = 250.0
_ILIGO_LASER_WAVELENGTH = 1.064e-6
_ILIGO_FINESSE = 220.0
_ILIGO_MIRROR_MASS = 11.0
_ILIGO_TEMPERATURE = 290.0
_ILIGO_STACK_FREQUENCY = 10.0
_ILIGO_SUSPENSION_FREQUENCY = 0.76
_ILIGO_SUSPENSION_QUALITY = 1e6
_ILIGO_COATING_FREQUENCY = 1e4
_ILIGO_COATING_QUALITY = 1e6

# ``laser_power``, ``srm_transmittance``, ``detuning_deg``, and ``zeta_deg``
# from the corresponding XLALSimNoisePSDQuantum wrapper in LALSimulation.
_ALIGO_CONFIGURATIONS = {
    "NoSRMLowPower": (25.0, 1.0, 0.0, 130.0),
    "NoSRMHighPower": (125.0, 1.0, 0.0, 130.0),
    "ZeroDetLowPower": (25.0, 0.2, 0.0, 116.0),
    "ZeroDetHighPower": (125.0, 0.2, 0.0, 116.0),
    "NSNSOpt": (125.0, 0.2, 11.0, 103.0),
    "BHBH20Deg": (20.0, 0.2, 20.0, 105.0),
    "HighFrequency": (125.0, 0.011, 4.7, 128.0),
}
_ALIGO_QUANTUM_MODELS = {
    f"aLIGOQuantum{name}": configuration
    for name, configuration in _ALIGO_CONFIGURATIONS.items()
}
_ALIGO_COMBINED_MODELS = {
    f"aLIGO{name}": configuration
    for name, configuration in _ALIGO_CONFIGURATIONS.items()
}
ALIGO_TORCH_ANALYTICAL_MODELS = frozenset(
    (*_ALIGO_QUANTUM_MODELS, *_ALIGO_COMBINED_MODELS, "aLIGOThermal")
)
GROUND_FIT_TORCH_ANALYTICAL_MODELS = frozenset(
    ("Virgo", "GEO", "GEOHF", "TAMA", "KAGRA", "AdvVirgo")
)
ILIGO_TORCH_ANALYTICAL_MODELS = frozenset(
    (
        "iLIGOSRD",
        "iLIGOSeismic",
        "iLIGOThermal",
        "iLIGOShot",
        "eLIGOShot",
        "iLIGOModel",
        "eLIGOModel",
    )
)
_P1200087_DATA_FILES = {
    "aLIGOEarlyLowSensitivityP1200087": (
        "LIGO-P1200087-v18-aLIGO_EARLY_LOW.txt"
    ),
    "aLIGOEarlyHighSensitivityP1200087": (
        "LIGO-P1200087-v18-aLIGO_EARLY_HIGH.txt"
    ),
    "aLIGOMidLowSensitivityP1200087": (
        "LIGO-P1200087-v18-aLIGO_MID_LOW.txt"
    ),
    "aLIGOMidHighSensitivityP1200087": (
        "LIGO-P1200087-v18-aLIGO_MID_HIGH.txt"
    ),
    "aLIGOLateLowSensitivityP1200087": (
        "LIGO-P1200087-v18-aLIGO_LATE_LOW.txt"
    ),
    "aLIGOLateHighSensitivityP1200087": (
        "LIGO-P1200087-v18-aLIGO_LATE_HIGH.txt"
    ),
    "aLIGODesignSensitivityP1200087": (
        "LIGO-P1200087-v18-aLIGO_DESIGN.txt"
    ),
    "aLIGOBNSOptimizedSensitivityP1200087": (
        "LIGO-P1200087-v18-aLIGO_BNS_OPTIMIZED.txt"
    ),
    "AdVEarlyLowSensitivityP1200087": (
        "LIGO-P1200087-v18-AdV_EARLY_LOW.txt"
    ),
    "AdVEarlyHighSensitivityP1200087": (
        "LIGO-P1200087-v18-AdV_EARLY_HIGH.txt"
    ),
    "AdVMidLowSensitivityP1200087": (
        "LIGO-P1200087-v18-AdV_MID_LOW.txt"
    ),
    "AdVMidHighSensitivityP1200087": (
        "LIGO-P1200087-v18-AdV_MID_HIGH.txt"
    ),
    "AdVLateLowSensitivityP1200087": (
        "LIGO-P1200087-v18-AdV_LATE_LOW.txt"
    ),
    "AdVLateHighSensitivityP1200087": (
        "LIGO-P1200087-v18-AdV_LATE_HIGH.txt"
    ),
    "AdVDesignSensitivityP1200087": (
        "LIGO-P1200087-v18-AdV_DESIGN.txt"
    ),
    "AdVBNSOptimizedSensitivityP1200087": (
        "LIGO-P1200087-v18-AdV_BNS_OPTIMIZED.txt"
    ),
}
_VERSIONED_DATA_FILES = {
    **_P1200087_DATA_FILES,
    "aLIGONoSRMLowPowerGWINC": "LIGO-T0900288-v3-NO_SRM.txt",
    "aLIGOZeroDetLowPowerGWINC": "LIGO-T0900288-v3-ZERO_DET_low_P.txt",
    "aLIGOZeroDetHighPowerGWINC": "LIGO-T0900288-v3-ZERO_DET_high_P.txt",
    "aLIGONSNSOptGWINC": "LIGO-T0900288-v3-NSNS_Opt.txt",
    "aLIGOBHBH20DegGWINC": "LIGO-T0900288-v3-BHBH_20deg.txt",
    "aLIGOHighFrequencyGWINC": "LIGO-T0900288-v3-High_Freq.txt",
    "CosmicExplorerP1600143": "LIGO-P1600143-v18-CE.txt",
    "CosmicExplorerPessimisticP1600143": (
        "LIGO-P1600143-v18-CE_Pessimistic.txt"
    ),
    "CosmicExplorerWidebandP1600143": (
        "LIGO-P1600143-v18-CE_Wideband.txt"
    ),
    "EinsteinTelescopeP1600143": "LIGO-P1600143-v18-ET_D.txt",
    "KAGRAOpeningSensitivityT1600593": (
        "LIGO-T1600593-v1-KAGRA_Opening.txt"
    ),
    "KAGRAEarlySensitivityT1600593": (
        "LIGO-T1600593-v1-KAGRA_Early.txt"
    ),
    "KAGRAMidSensitivityT1600593": "LIGO-T1600593-v1-KAGRA_Mid.txt",
    "KAGRALateSensitivityT1600593": "LIGO-T1600593-v1-KAGRA_Late.txt",
    "KAGRADesignSensitivityT1600593": (
        "LIGO-T1600593-v1-KAGRA_Design.txt"
    ),
    "aLIGOAPlusDesignSensitivityT1800042": (
        "LIGO-T1800042-v5-aLIGO_APLUS.txt"
    ),
    "aLIGODesignSensitivityT1800044": (
        "LIGO-T1800044-v5-aLIGO_DESIGN.txt"
    ),
    "aLIGOaLIGODesignSensitivityT1800044": (
        "LIGO-T1800044-v5-aLIGO_DESIGN.txt"
    ),
    "aLIGOO3LowT1800545": "LIGO-T1800545-v1-aLIGO_O3low.txt",
    "aLIGOaLIGOO3LowT1800545": "LIGO-T1800545-v1-aLIGO_O3low.txt",
    "aLIGO140MpcT1800545": "LIGO-T1800545-v1-aLIGO_140Mpc.txt",
    "aLIGOaLIGO140MpcT1800545": "LIGO-T1800545-v1-aLIGO_140Mpc.txt",
    "aLIGO175MpcT1800545": "LIGO-T1800545-v1-aLIGO_175Mpc.txt",
    "aLIGOaLIGO175MpcT1800545": "LIGO-T1800545-v1-aLIGO_175Mpc.txt",
    "AdVO4IntermediateT1800545": (
        "LIGO-T1800545-v1-AdV_O4intermediate.txt"
    ),
    "aLIGOAdVO4IntermediateT1800545": (
        "LIGO-T1800545-v1-AdV_O4intermediate.txt"
    ),
    "AdVO4T1800545": "LIGO-T1800545-v1-AdV_O4.txt",
    "aLIGOAdVO4T1800545": "LIGO-T1800545-v1-AdV_O4.txt",
    "AdVO3LowT1800545": "LIGO-T1800545-v1-AdV_O3low.txt",
    "aLIGOAdVO3LowT1800545": "LIGO-T1800545-v1-AdV_O3low.txt",
    "KAGRA128MpcT1800545": "LIGO-T1800545-v1-KAGRA_128Mpc.txt",
    "aLIGOKAGRA128MpcT1800545": "LIGO-T1800545-v1-KAGRA_128Mpc.txt",
    "KAGRA25MpcT1800545": "LIGO-T1800545-v1-KAGRA_25Mpc.txt",
    "aLIGOKAGRA25MpcT1800545": "LIGO-T1800545-v1-KAGRA_25Mpc.txt",
    "KAGRA80MpcT1800545": "LIGO-T1800545-v1-KAGRA_80Mpc.txt",
    "aLIGOKAGRA80MpcT1800545": "LIGO-T1800545-v1-KAGRA_80Mpc.txt",
}
P1200087_TORCH_ANALYTICAL_MODELS = frozenset(_P1200087_DATA_FILES)
DATA_FILE_TORCH_ANALYTICAL_MODELS = frozenset(_VERSIONED_DATA_FILES)
TORCH_ANALYTICAL_PSD_MODELS = (
    ALIGO_TORCH_ANALYTICAL_MODELS
    | GROUND_FIT_TORCH_ANALYTICAL_MODELS
    | ILIGO_TORCH_ANALYTICAL_MODELS
    | DATA_FILE_TORCH_ANALYTICAL_MODELS
)


class _DataFileNativeUnsupported(Exception):
    """Signal that a resolved table must use LALSimulation's reader."""


def _magnitude_squared(values):
    """Match the squared ``cabs`` terms in LALSimulation's expression."""
    if values.is_complex():
        return torch.view_as_real(values).square().sum(dim=-1)
    return values.square()


def _iligo_srd(frequencies):
    """Evaluate the initial-LIGO Science Requirements Document fit."""
    seismic_amplitude = 1.57271
    seismic_exponent = -14.0
    thermal_amplitude = 3.80591e-19
    thermal_exponent = -2.0
    shot_amplitude = 1.12277e-23
    shot_frequency = 89.3676

    seismic = (
        seismic_amplitude
        * seismic_amplitude
        * frequencies.pow(2.0 * seismic_exponent)
    )
    thermal = (
        thermal_amplitude
        * thermal_amplitude
        * frequencies.pow(2.0 * thermal_exponent)
    )
    shot = shot_amplitude * shot_amplitude * (
        1.0 + (frequencies / shot_frequency).pow(2.0)
    )
    return seismic + thermal + shot


def _iligo_seismic(frequencies):
    """Evaluate LALSimulation's initial-LIGO seismic component."""
    pendulum = (_ILIGO_SUSPENSION_FREQUENCY / frequencies).pow(2.0)
    stack = (_ILIGO_STACK_FREQUENCY / frequencies).pow(8.0)
    ground = torch.full_like(frequencies, 1e-18)
    ground = torch.where(
        frequencies > 10.0,
        ground * (10.0 / frequencies).pow(4.0),
        ground,
    )
    return ground * (pendulum * stack / _ILIGO_ARM_LENGTH).pow(2.0)


def _iligo_thermal(frequencies):
    """Evaluate the initial-LIGO suspension and coating thermal fit."""
    suspension_scale = (
        2.0
        * lal.K_SI
        * _ILIGO_TEMPERATURE
        / (
            _ILIGO_ARM_LENGTH
            * _ILIGO_ARM_LENGTH
            * _ILIGO_MIRROR_MASS
            * _ILIGO_SUSPENSION_QUALITY
            * (lal.PI * _ILIGO_SUSPENSION_FREQUENCY) ** 3
        )
    )
    coating_scale = (
        2.0
        * lal.K_SI
        * _ILIGO_TEMPERATURE
        / (
            _ILIGO_ARM_LENGTH
            * _ILIGO_ARM_LENGTH
            * _ILIGO_MIRROR_MASS
            * _ILIGO_COATING_QUALITY
            * (lal.PI * _ILIGO_COATING_FREQUENCY) ** 3
        )
    )
    suspension = suspension_scale * (
        _ILIGO_SUSPENSION_FREQUENCY / frequencies
    ).pow(5.0)
    coating = coating_scale * (_ILIGO_COATING_FREQUENCY / frequencies)
    return suspension + coating


def _first_generation_shot(frequencies, efficiency):
    """Evaluate LALSimulation's conventional first-generation shot noise."""
    storage_time = (
        _ILIGO_ARM_LENGTH
        * _ILIGO_FINESSE
        / (lal.PI * lal.C_SI)
    )
    pole_frequency = 1.0 / (4.0 * lal.PI * storage_time)
    dc_noise = (
        lal.PI
        * lal.HBAR_SI
        * _ILIGO_LASER_WAVELENGTH
        * pole_frequency
        * pole_frequency
        / (
            lal.C_SI
            * efficiency
            * _ILIGO_LASER_POWER_BS
        )
    )
    sensing = (
        torch.exp(
            2.0
            * lal.PI
            * 1j
            * frequencies
            * _ILIGO_ARM_LENGTH
            / lal.C_SI
        )
        * math.sinh(
            2.0
            * lal.PI
            * pole_frequency
            * _ILIGO_ARM_LENGTH
            / lal.C_SI
        )
        / torch.sinh(
            (
                2.0
                * lal.PI
                * pole_frequency
                * _ILIGO_ARM_LENGTH
                / lal.C_SI
            )
            * (1.0 + 1j * (frequencies / pole_frequency))
        )
    )
    return dc_noise / _magnitude_squared(sensing)


def _first_generation_detector_psd(psd_name, frequencies):
    """Dispatch initial- and enhanced-LIGO component/model expressions."""
    if psd_name == "iLIGOSRD":
        return _iligo_srd(frequencies)
    if psd_name == "iLIGOSeismic":
        return _iligo_seismic(frequencies)
    if psd_name == "iLIGOThermal":
        return _iligo_thermal(frequencies)
    if psd_name == "iLIGOShot":
        return _first_generation_shot(frequencies, 0.9 / 3.0)
    if psd_name == "eLIGOShot":
        return _first_generation_shot(frequencies, 0.9)
    if psd_name not in ("iLIGOModel", "eLIGOModel"):
        raise ValueError(f"Unsupported Torch i/eLIGO PSD {psd_name}")

    seismic = _iligo_seismic(frequencies)
    thermal = _iligo_thermal(frequencies)
    efficiency = 0.9 / 3.0 if psd_name == "iLIGOModel" else 0.9
    shot = _first_generation_shot(frequencies, efficiency)
    return shot + seismic + thermal


def _ground_detector_fit(psd_name, frequencies):
    """Evaluate LALSimulation's data-free detector PSD fits."""
    if psd_name == "Virgo":
        x = frequencies / 500.0
        return 10.2e-46 * (
            (7.87 * x).pow(-4.8)
            + (6.0 / 17.0) / x
            + 1.0
            + x.square()
        )
    if psd_name == "GEO":
        x = frequencies / 150.0
        x2 = x.square()
        seismic = 1e-16 * x.pow(-30.0)
        thermal = 34.0 / x
        shot = 20.0 * (1.0 - x2 + 0.5 * x2.square()) / (1.0 + 0.5 * x2)
        return 1e-46 * (seismic + thermal + shot)
    if psd_name == "GEOHF":
        f2 = frequencies.square()
        return (
            7.18e-46 * (1.0 + f2 / (1059.0 * 1059.0))
            + 4.90e-41 / f2
            + 8.91e-43 / frequencies
            + 1.6e-17 / frequencies.pow(16.0)
        )
    if psd_name == "TAMA":
        x = frequencies / 400.0
        return 75e-46 * (
            x.pow(-5.0) + 13.0 / x + 9.0 * (1.0 + x.square())
        )
    if psd_name == "KAGRA":
        x = torch.log(frequencies / 100.0)
        x2 = x.square()
        asd = 6.499e-25 * (
            9.72e-9 * torch.exp(-1.43 - 9.88 * x - 0.23 * x2)
            + 1.17 * torch.exp(0.14 - 3.10 * x - 0.26 * x2)
            + 1.70 * torch.exp(0.14 + 1.09 * x - 0.013 * x2)
            + 1.25 * torch.exp(0.071 + 2.83 * x - 4.91 * x2)
        )
        return asd.square()

    if psd_name == "AdvVirgo":
        x = torch.log(frequencies / 300.0)
        x2 = x.square()
        asd = 1.259e-24 * (
            0.07 * torch.exp(-0.142 - 1.437 * x + 0.407 * x2)
            + 3.1 * torch.exp(-0.466 - 1.043 * x - 0.548 * x2)
            + 0.4 * torch.exp(-0.304 + 2.896 * x - 0.293 * x2)
            + 0.09 * torch.exp(1.466 + 3.722 * x - 0.984 * x2)
        )
        return asd.square()
    raise ValueError(f"Unsupported Torch ground-detector PSD fit {psd_name}")


def _aligo_quantum(
    frequencies,
    laser_power,
    srm_transmittance,
    detuning_deg,
    zeta_deg,
):
    """Port LALSimulation's shared advanced-LIGO quantum expression."""
    eta = 0.9
    detuning = detuning_deg * lal.PI_180
    zeta = zeta_deg * lal.PI_180

    omega = 2.0 * lal.PI * frequencies
    laser_omega = 2.0 * lal.PI * lal.C_SI / _LASER_WAVELENGTH
    signal_recycling_loss = _BEAM_SPLITTER_LOSS
    photodiode_loss = 1.0 - eta
    tau = math.sqrt(srm_transmittance)
    rho = math.sqrt(1.0 - tau * tau)
    phi = (lal.PI - detuning) / 2.0
    arm_loss = 2.0 * _MIRROR_LOSS
    gamma_ac = _ITM_TRANSMITTANCE * lal.C_SI / (4.0 * _ARM_LENGTH)
    epsilon = arm_loss / (2.0 * gamma_ac * _ARM_LENGTH / lal.C_SI)
    itm_reflectivity = math.sqrt(1.0 - _ITM_TRANSMITTANCE)
    arm_reflectivity = (
        itm_reflectivity
        - _ITM_TRANSMITTANCE
        * math.sqrt(1.0 - 2.0 * _MIRROR_LOSS)
        / (
            1.0
            - itm_reflectivity
            * math.sqrt(1.0 - 2.0 * _MIRROR_LOSS)
        )
    )
    recycling_gain = _PRM_TRANSMITTANCE / (
        1.0
        + math.sqrt(1.0 - _PRM_TRANSMITTANCE)
        * arm_reflectivity
        * math.sqrt(1.0 - _BEAM_SPLITTER_LOSS)
    ) ** 2
    beam_splitter_power = recycling_gain * laser_power
    sql_power = (
        _MIRROR_MASS
        * _ARM_LENGTH
        * _ARM_LENGTH
        * gamma_ac ** 4
        / (4.0 * laser_omega)
    )
    kappa = (
        2.0
        * (beam_splitter_power / sql_power)
        * gamma_ac ** 4
        / (omega * omega * (gamma_ac * gamma_ac + omega * omega))
    )
    beta = torch.atan(omega / gamma_ac)
    h_sql = torch.sqrt(
        8.0
        * lal.HBAR_SI
        / (_MIRROR_MASS * (omega * _ARM_LENGTH) ** 2)
    )

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_two_phi = math.sin(2.0 * phi)
    cos_two_phi = math.cos(2.0 * phi)
    cos_beta = torch.cos(beta)
    cos_two_beta = torch.cos(2.0 * beta)
    sqrt_efficiency = math.sqrt(1.0 - photodiode_loss)
    exp_two_i_beta = torch.exp(2j * beta)

    # Readout denominator (D1 and D2 in Buonanno and Chen, equation 5.13).
    exp_four_i_beta = torch.exp(4j * beta)
    d1 = sqrt_efficiency * (
        -(1.0 + rho * exp_two_i_beta) * sin_phi
        + 0.25
        * epsilon
        * (
            3.0
            + rho
            + 2.0 * rho * exp_four_i_beta
            + exp_two_i_beta * (1.0 + 5.0 * rho)
        )
        * sin_phi
        + 0.5
        * signal_recycling_loss
        * exp_two_i_beta
        * rho
        * sin_phi
    )
    d2 = sqrt_efficiency * (
        -(-1.0 + rho * exp_two_i_beta) * cos_phi
        + 0.25
        * epsilon
        * (
            -3.0
            + rho
            + 2.0 * rho * exp_four_i_beta
            + exp_two_i_beta * (-1.0 + 5.0 * rho)
        )
        * cos_phi
        + 0.5
        * signal_recycling_loss
        * exp_two_i_beta
        * rho
        * cos_phi
    )
    del exp_four_i_beta
    sin_zeta = math.sin(zeta)
    cos_zeta = math.cos(zeta)
    denominator = _magnitude_squared(d1 * sin_zeta + d2 * cos_zeta)
    del d1, d2

    # C coefficients. Accumulate terms as they are formed to keep the peak
    # complex-tensor footprint bounded for long PSDs.
    c11 = sqrt_efficiency * (
        (1.0 + rho * rho)
        * (cos_two_phi + kappa / 2.0 * sin_two_phi)
        - 2.0 * rho * cos_two_beta
        - 0.25
        * epsilon
        * (
            -2.0
            * (1.0 + exp_two_i_beta)
            * (1.0 + exp_two_i_beta)
            * rho
            + 4.0
            * (1.0 + rho * rho)
            * cos_beta.square()
            * cos_two_phi
            + (3.0 + exp_two_i_beta)
            * kappa
            * (1.0 + rho * rho)
            * sin_two_phi
        )
        + signal_recycling_loss
        * (
            exp_two_i_beta * rho
            - 0.5
            * (1.0 + rho * rho)
            * (cos_two_phi + kappa / 2.0 * sin_two_phi)
        )
    )
    c21 = sqrt_efficiency * tau * tau * (
        sin_two_phi
        - kappa * cos_phi ** 2
        + 0.5
        * epsilon
        * cos_phi
        * (
            (3.0 + exp_two_i_beta) * kappa * sin_phi
            - 4.0 * cos_beta.square() * sin_phi
        )
        + 0.5
        * signal_recycling_loss
        * (-sin_two_phi + kappa * cos_phi ** 2)
    )
    numerator = _magnitude_squared(c11 * sin_zeta + c21 * cos_zeta)
    del c21
    c12 = sqrt_efficiency * tau * tau * (
        -(sin_two_phi + kappa * sin_phi ** 2)
        + 0.5
        * epsilon
        * sin_phi
        * (
            (3.0 + exp_two_i_beta) * kappa * sin_phi
            + 4.0 * cos_beta.square() * cos_phi
        )
        + 0.5
        * signal_recycling_loss
        * (sin_two_phi + kappa * sin_phi ** 2)
    )
    numerator.add_(
        _magnitude_squared(c12 * sin_zeta + c11 * cos_zeta)
    )
    del c11, c12

    # P coefficients.
    sqrt_signal_loss = math.sqrt(signal_recycling_loss)
    p11 = (
        0.5
        * sqrt_efficiency
        * sqrt_signal_loss
        * tau
        * (
            -2.0 * rho * exp_two_i_beta
            + 2.0 * cos_two_phi
            + kappa * sin_two_phi
        )
    )
    p21 = (
        sqrt_efficiency
        * sqrt_signal_loss
        * tau
        * cos_phi
        * (2.0 * sin_phi - kappa * cos_phi)
    )
    numerator.add_(
        _magnitude_squared(p11 * sin_zeta + p21 * cos_zeta)
    )
    del p21
    p12 = (
        -sqrt_efficiency
        * sqrt_signal_loss
        * tau
        * sin_phi
        * (2.0 * cos_phi + kappa * sin_phi)
    )
    numerator.add_(
        _magnitude_squared(p12 * sin_zeta + p11 * cos_zeta)
    )
    del p11, p12

    # Q coefficients. Q12 and Q21 are identically zero.
    exp_minus_two_i_beta = torch.exp(-2j * beta)
    q11 = math.sqrt(photodiode_loss) * (
        exp_minus_two_i_beta
        + rho * rho * exp_two_i_beta
        - rho * (2.0 * cos_two_phi + kappa * sin_two_phi)
        + 0.5
        * epsilon
        * rho
        * (
            exp_minus_two_i_beta * cos_two_phi
            + exp_two_i_beta
            * (
                -2.0 * rho
                - 2.0 * rho * cos_two_beta
                + cos_two_phi
                + kappa * sin_two_phi
            )
            + 2.0 * cos_two_phi
            + 3.0 * kappa * sin_two_phi
        )
        - 0.5
        * signal_recycling_loss
        * rho
        * (
            2.0 * rho * exp_two_i_beta
            - 2.0 * cos_two_phi
            - kappa * sin_two_phi
        )
    )
    del exp_minus_two_i_beta
    numerator.add_(_magnitude_squared(q11 * sin_zeta))
    numerator.add_(_magnitude_squared(q11 * cos_zeta))
    del q11

    # N coefficients.
    exp_i_beta = torch.exp(1j * beta)
    exp_minus_i_beta = torch.exp(-1j * beta)
    n11 = (
        sqrt_efficiency
        * math.sqrt(epsilon / 2.0)
        * tau
        * (
            kappa * (1.0 + rho * exp_two_i_beta) * sin_phi
            + 2.0
            * cos_beta
            * (
                exp_minus_i_beta * cos_phi
                - rho
                * exp_i_beta
                * (cos_phi + kappa * sin_phi)
            )
        )
    )
    n21 = (
        sqrt_efficiency
        * math.sqrt(2.0 * epsilon)
        * tau
        * (
            -kappa * (1.0 + rho) * cos_phi
            + 2.0
            * cos_beta
            * (exp_minus_i_beta + rho * exp_i_beta)
            * cos_beta
            * sin_phi
        )
    )
    numerator.add_(
        _magnitude_squared(n11 * sin_zeta + n21 * cos_zeta)
    )
    del n11, n21
    n12 = (
        -sqrt_efficiency
        * math.sqrt(2.0 * epsilon)
        * tau
        * (exp_minus_i_beta + rho * exp_i_beta)
        * cos_beta
        * sin_phi
    )
    n22 = (
        -sqrt_efficiency
        * math.sqrt(2.0 * epsilon)
        * tau
        * (-exp_minus_i_beta + rho * exp_i_beta)
        * cos_beta
        * cos_phi
    )
    numerator.add_(
        _magnitude_squared(n12 * sin_zeta + n22 * cos_zeta)
    )

    return (
        h_sql.square()
        / (2.0 * kappa * tau * tau * denominator)
        * numerator
    )


def _aligo_thermal(frequencies):
    """Return LALSimulation's suspension plus coating thermal fit."""
    suspension_scale = (
        2.0
        * lal.K_SI
        * _TEMPERATURE
        / (
            _ARM_LENGTH
            * _ARM_LENGTH
            * _MIRROR_MASS
            * _SUSPENSION_QUALITY
            * (lal.PI * _SUSPENSION_FREQUENCY) ** 3
        )
    )
    coating_scale = (
        2.0
        * lal.K_SI
        * _TEMPERATURE
        / (
            _ARM_LENGTH
            * _ARM_LENGTH
            * _MIRROR_MASS
            * _COATING_QUALITY
            * (lal.PI * _COATING_FREQUENCY) ** 3
        )
    )
    return (
        suspension_scale
        * (_SUSPENSION_FREQUENCY / frequencies) ** 5
        + coating_scale * (_COATING_FREQUENCY / frequencies)
    )


def _read_data_file_asd(psd_name):
    """Resolve and read one of LALSimulation's versioned ASD tables."""
    filename = _VERSIONED_DATA_FILES[psd_name]
    path = _resolve_data_file(filename)
    if path is None:
        raise RuntimeError(
            f"Unable to resolve LALSimulation PSD data file {filename}"
        )
    rows = []
    try:
        with open(path, encoding="ascii", newline="") as table_file:
            for line in table_file:
                # LAL's fixed-buffer reader requires a newline on every line,
                # including comments.  Use a conservative line-size envelope
                # so longer resolver overrides take the exact LAL fallback on
                # every supported platform.
                if not line.endswith("\n") or len(line) >= 1024:
                    raise _DataFileNativeUnsupported
                if line.startswith("#"):
                    continue
                columns = line.split()
                if len(columns) != 2 or "_" in line:
                    raise _DataFileNativeUnsupported
                try:
                    rows.append((float(columns[0]), float(columns[1])))
                except ValueError as exc:
                    raise _DataFileNativeUnsupported from exc
    except OSError as exc:
        raise RuntimeError(
            f"Invalid LALSimulation PSD data file {filename}"
        ) from exc
    except UnicodeError as exc:
        raise _DataFileNativeUnsupported from exc

    table = numpy.asarray(rows, dtype=numpy.float64)
    if (
        table.ndim != 2
        or table.shape[0] < 3
        or table.shape[1] != 2
        or not numpy.all(numpy.isfinite(table))
        or not numpy.all(table > 0.0)
        or not numpy.all(numpy.diff(table[:, 0]) > 0.0)
    ):
        raise _DataFileNativeUnsupported
    return (
        numpy.ascontiguousarray(table[:, 0]),
        numpy.ascontiguousarray(table[:, 1]),
    )


def _resolve_data_file(filename):
    """Resolve LAL data without making core LAL a Torch dependency.

    The real resolver remains authoritative when LAL is installed.  Without
    it, reproduce the useful public search surface: the current directory,
    ``LAL_DATA_PATH`` entries, and data shipped in the ``lalapps`` wheel.
    """
    if hasattr(lal, "FileResolvePath"):
        return lal.FileResolvePath(filename)

    candidates = [filename]
    candidates.extend(
        os.path.join(directory, filename)
        for directory in os.environ.get("LAL_DATA_PATH", "").split(
            os.pathsep
        )
        if directory
    )
    try:
        spec = importlib.util.find_spec("lalapps")
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations is not None:
        candidates.extend(
            os.path.join(directory, "data", filename)
            for directory in spec.submodule_search_locations
        )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _data_file_psd(psd_name, length, delta_f, low_freq_cutoff, device):
    """Match ``XLALSimNoisePSDFromFile`` for a bundled versioned table."""
    frequency_data, asd_data = _read_data_file_asd(psd_name)
    frequencies = torch.as_tensor(
        frequency_data,
        dtype=torch.float64,
        device=device,
    )
    asd = torch.as_tensor(
        asd_data,
        dtype=torch.float64,
        device=device,
    )
    log_asd = torch.where(asd > 0.0, torch.log(asd), torch.zeros_like(asd))
    values = torch.zeros(length, dtype=torch.float64, device=device)

    # XLALSimNoisePSDFromFile uses zero as both an index and an "unset"
    # sentinel while locating the first positive ASD sample.  Every supported
    # bundled table starts positive, so its effective first valid row is row
    # one and interpolation begins with the row-one/row-two interval.  Derive
    # the index with the same sentinel behavior so a resolver override retains
    # LAL's semantics too.
    positive = numpy.flatnonzero(asd_data > 0.0)
    if positive.size == 0:
        first_valid = 0
    elif positive[0] == 0 and positive.size > 1:
        first_valid = int(positive[1])
    else:
        first_valid = int(positive[0])
    flow = low_freq_cutoff
    if flow <= 0.0:
        flow = frequency_data[first_valid]
    kmin = int(flow / delta_f)
    if kmin == 0:
        kmin = 1

    if kmin < length - 1:
        sample_frequencies = (
            torch.arange(
                kmin,
                length - 1,
                dtype=torch.float64,
                device=device,
            )
            * delta_f
        )
        upper = torch.searchsorted(
            frequencies,
            sample_frequencies,
            right=False,
        )
        upper = torch.clamp(
            upper,
            first_valid + 1,
            frequencies.numel() - 1,
        )
        lower = upper - 1
        weight = (
            (frequencies[upper] - sample_frequencies)
            / (frequencies[upper] - frequencies[lower])
        )
        interpolated = (
            weight * log_asd[lower]
            + (1.0 - weight) * log_asd[upper]
        )
        values[kmin:-1] = torch.exp(2.0 * interpolated)
    return values


def analytical_psd(
    psd_name,
    length,
    delta_f,
    device,
    low_freq_cutoff=0.0,
):
    """Generate a supported LAL analytical PSD on ``device``."""
    if psd_name not in TORCH_ANALYTICAL_PSD_MODELS:
        raise ValueError(f"Unsupported Torch analytical PSD {psd_name}")
    if device.type == "mps":
        raise TypeError(
            f"{psd_name} requires float64; Torch MPS only supports "
            "float32, which underflows its physical PSD values"
        )

    if psd_name in DATA_FILE_TORCH_ANALYTICAL_MODELS:
        values = _data_file_psd(
            psd_name,
            length,
            delta_f,
            low_freq_cutoff,
            device,
        )
        return FrequencySeries(
            TorchArrayData(values),
            delta_f=delta_f,
            copy=False,
        )

    values = torch.zeros(length, dtype=torch.float64, device=device)
    if length > 2:
        frequencies = (
            torch.arange(
                1,
                length - 1,
                dtype=torch.float64,
                device=device,
            )
            * delta_f
        )
        if psd_name in ILIGO_TORCH_ANALYTICAL_MODELS:
            values[1:-1] = _first_generation_detector_psd(
                psd_name,
                frequencies,
            )
        elif psd_name in GROUND_FIT_TORCH_ANALYTICAL_MODELS:
            values[1:-1] = _ground_detector_fit(psd_name, frequencies)
        elif psd_name == "aLIGOThermal":
            values[1:-1] = _aligo_thermal(frequencies)
        else:
            configuration = _ALIGO_QUANTUM_MODELS.get(psd_name)
            include_thermal = configuration is None
            if include_thermal:
                configuration = _ALIGO_COMBINED_MODELS[psd_name]
            values[1:-1] = _aligo_quantum(
                frequencies,
                *configuration,
            )
            if include_thermal:
                values[1:-1].add_(_aligo_thermal(frequencies))

    return FrequencySeries(
        TorchArrayData(values),
        delta_f=delta_f,
        copy=False,
    )


def aligo_analytical_psd(psd_name, length, delta_f, device):
    """Generate a LAL analytical advanced-LIGO PSD on ``device``."""
    if psd_name not in ALIGO_TORCH_ANALYTICAL_MODELS:
        raise ValueError(f"Unsupported Torch analytical aLIGO PSD {psd_name}")
    return analytical_psd(psd_name, length, delta_f, device)


def aligo_zero_det_high_power(length, delta_f, device):
    """Generate the LAL aLIGO zero-detuned, high-power PSD on ``device``."""
    return aligo_analytical_psd(
        "aLIGOZeroDetHighPower",
        length,
        delta_f,
        device,
    )
