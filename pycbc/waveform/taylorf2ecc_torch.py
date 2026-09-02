# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Native Torch implementation of the aligned-spin TaylorF2Ecc waveform.

The circular phasing, amplitude, and polarization construction are shared
with :mod:`pycbc.waveform.taylorf2_torch`.  This module ports LALSuite's
low-eccentricity phase correction through relative 3PN order.  All
frequency-dependent evaluation is performed on the active Torch device.
"""

import math

from pycbc import lal_compat as lal

import pycbc.scheme as _scheme
from pycbc.waveform._rom_hybrid_torch import _minimum_sequence_frequency

from .taylorf2_torch import (
    _as_order,
    _contact_frequency,
    _taylorf2_inputs,
    _taylorf2_polarizations,
    _taylorf2_samples,
    taylorf2_native_supported,
    taylorf2_sequence_native_supported,
)


_ECCENTRICITY_ORDERS = frozenset((-1, 0, 1, 2, 3, 4, 5, 6))
_MPS_MAX_ECCENTRIC_PHASE = 5.0e4


def _circular_params(params):
    """Return TaylorF2 carrier parameters for an eccentric request."""

    circular = dict(params)
    circular["eccentricity"] = 0.0
    circular["eccentricity_order"] = -1
    return circular


def _native_features_supported(params):
    """Return whether the eccentric options are implemented here."""

    try:
        eccentricity = float(params.get("eccentricity", 0.0))
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0:
        return False
    if _as_order(params.get("eccentricity_order", -1)) not in (
        _ECCENTRICITY_ORDERS
    ):
        return False
    return True


def taylorf2ecc_native_supported(params):
    """Return whether regular-grid TaylorF2Ecc generation is native."""

    return (
        _native_features_supported(params)
        and taylorf2_native_supported(_circular_params(params))
        and _native_device_supported(params, sequence=False)
    )


def taylorf2ecc_sequence_native_supported(params):
    """Return whether arbitrary-frequency TaylorF2Ecc is native."""

    return (
        _native_features_supported(params)
        and taylorf2_sequence_native_supported(_circular_params(params))
        and _native_device_supported(params, sequence=True)
    )


def _taylorf2ecc_sequence_frequencies(sample_points, inputs):
    """Validate sample points and move them to the active Torch device."""
    import torch

    from pycbc.types.array_torch import TorchArrayData

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError(
            "TaylorF2Ecc sample_points must be a non-empty vector"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("TaylorF2Ecc sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("TaylorF2Ecc sample_points must be positive")
    return frequencies


def _eccentric_phase_bracket(
    velocity,
    velocity0,
    eta,
    order,
    *,
    log,
    ones_like,
):
    """Evaluate the eccentric PN bracket for scalars or tensors."""
    eta2 = eta * eta
    eta3 = eta2 * eta
    v = velocity
    v0 = velocity0
    phase = ones_like(v)

    if order >= 2:
        phase = phase + (
            (29.9076223 / 8.1976608 + 18.766963 / 2.927736 * eta) * v**2
            + (2.833 / 1.008 - 19.7 / 3.6 * eta) * v0**2
        )
    if order >= 3:
        phase = phase + (
            (-28.19123 / 2.82600 * math.pi) * v**3
            + (37.7 / 7.2 * math.pi) * v0**3
        )
    if order >= 4:
        phase = phase + (
            (
                16.237683263 / 3.330429696
                + 241.33060753 / 9.71375328 * eta
                + 156.2608261 / 6.9383952 * eta2
            )
            * v**4
            + (
                84.7282939759 / 8.2632420864
                - 7.18901219 / 3.68894736 * eta
                - 36.97091711 / 1.05398496 * eta2
            )
            * v**2
            * v0**2
            + (
                -1.193251 / 3.048192
                - 66.317 / 9.072 * eta
                + 18.155 / 1.296 * eta2
            )
            * v0**4
        )
    if order >= 5:
        phase = phase + (
            (
                -28.31492681 / 1.18395270 * math.pi
                - 115.52066831 / 2.70617760 * math.pi * eta
            )
            * v**5
            + (
                -79.86575459 / 2.84860800 * math.pi
                + 55.5367231 / 1.0173600 * math.pi * eta
            )
            * v**3
            * v0**2
            + (
                112.751736071 / 5.902315776 * math.pi
                + 70.75145051 / 2.10796992 * math.pi * eta
            )
            * v**2
            * v0**3
            + (
                76.4881 / 9.0720 * math.pi
                - 94.9457 / 2.2680 * math.pi * eta
            )
            * v0**5
        )
    if order >= 6:
        coeff_v6 = (
            -436.03153867072577087 / 1.32658535116800000
            + 53.6803271 / 1.9782000 * lal.GAMMA
            + 157.22503703 / 3.25555200 * math.pi**2
            + (
                2991.72861614477 / 6.89135247360
                - 15.075413 / 1.446912 * math.pi**2
            )
            * eta
            + 345.5209264991 / 4.1019955200 * eta2
            + 506.12671711 / 8.78999040 * eta3
            + 384.3505163 / 5.9346000 * math.log(2.0)
            - 112.1397129 / 1.7584000 * math.log(3.0)
        )
        coeff_v4_v02 = (
            46.001356684079 / 3.357073133568
            + 253.471410141755 / 5.874877983744 * eta
            - 169.3852244423 / 2.3313007872 * eta2
            - 307.833827417 / 2.497822272 * eta3
        )
        coeff_v3_v03 = -106.2809371 / 2.0347200 * math.pi**2
        coeff_v2_v04 = (
            -3.56873002170973 / 2.49880440692736
            - 260.399751935005 / 8.924301453312 * eta
            + 15.0484695827 / 3.5413894656 * eta2
            + 340.714213265 / 3.794345856 * eta3
        )
        coeff_v06 = (
            265.31900578691 / 1.68991764480
            - 33.17 / 1.26 * lal.GAMMA
            + 12.2833 / 1.0368 * math.pi**2
            + (91.55185261 / 5.48674560 - 3.977 / 1.152 * math.pi**2) * eta
            - 5.732473 / 1.306368 * eta2
            - 30.90307 / 1.39968 * eta3
            + 87.419 / 1.890 * math.log(2.0)
            - 260.01 / 5.60 * math.log(3.0)
        )
        phase = phase + (
            (
                coeff_v6
                + 53.6803271 / 3.9564000 * log(16.0 * v**2)
            )
            * v**6
            + coeff_v4_v02 * v**4 * v0**2
            + coeff_v3_v03 * v**3 * v0**3
            + coeff_v2_v04 * v**2 * v0**4
            + (
                coeff_v06
                - 33.17 / 2.52 * log(16.0 * v0**2)
            )
            * v0**6
        )

    return phase


def _eccentric_phase_polynomial(velocity, velocity0, eccentricity, eta, order):
    """Evaluate LAL's eccentric correction before the overall ``v^-5``."""
    import torch

    if eccentricity == 0.0:
        return torch.zeros_like(velocity)
    if order == -1:
        order = 6

    phase = _eccentric_phase_bracket(
        velocity,
        velocity0,
        eta,
        order,
        log=torch.log,
        ones_like=torch.ones_like,
    )
    v = velocity
    v0 = velocity0

    global_factor = (
        -2.355
        / 1.462
        * eccentricity**2
        * torch.pow(v0 / v, 19.0 / 3.0)
        * (3.0 / (128.0 * eta))
    )
    return phase * global_factor


def _eccentric_phase_scalar(
    frequency,
    f_ecc,
    total_mass,
    eta,
    eccentricity,
    order,
):
    """Evaluate the complete eccentric phase with Python scalar arithmetic."""

    if eccentricity == 0.0:
        return 0.0
    if order == -1:
        order = 6

    pi_mass = math.pi * total_mass * lal.MTSUN_SI
    velocity = (pi_mass * frequency) ** (1.0 / 3.0)
    velocity0 = (pi_mass * f_ecc) ** (1.0 / 3.0)
    phase = _eccentric_phase_bracket(
        velocity,
        velocity0,
        eta,
        order,
        log=math.log,
        ones_like=lambda _value: 1.0,
    )
    global_factor = (
        -2.355
        / 1.462
        * eccentricity**2
        * (velocity0 / velocity) ** (19.0 / 3.0)
        * (3.0 / (128.0 * eta))
    )
    return phase * global_factor / velocity**5


def _native_device_supported(params, *, sequence):
    """Bound the extra eccentric-phase roundoff on Apple MPS."""

    state = _scheme.mgr.state
    if not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    ):
        return True

    try:
        eccentricity = float(params.get("eccentricity", 0.0))
        if eccentricity == 0.0:
            return True
        mass1 = float(params["mass1"])
        mass2 = float(params["mass2"])
        total_mass = mass1 + mass2
        eta = mass1 * mass2 / total_mass**2
        if sequence:
            start_frequency = _minimum_sequence_frequency(
                params["sample_points"]
            )
        else:
            start_frequency = float(params["f_lower"])
        reference_frequency = float(params.get("f_ref", 0.0) or 0.0)
        order = _as_order(params.get("eccentricity_order", -1))
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
        ZeroDivisionError,
    ):
        return False

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (total_mass, eta, start_frequency)
    ):
        return False
    if not math.isfinite(reference_frequency) or reference_frequency < 0.0:
        return False

    f_ecc = reference_frequency or start_frequency
    phase_frequencies = [start_frequency]
    if reference_frequency > 0.0:
        phase_frequencies.append(reference_frequency)
    try:
        phase_scale = max(
            abs(
                _eccentric_phase_scalar(
                    frequency,
                    f_ecc,
                    total_mass,
                    eta,
                    eccentricity,
                    order,
                )
            )
            for frequency in phase_frequencies
        )
    except (ValueError, OverflowError, ZeroDivisionError):
        return False
    return (
        math.isfinite(phase_scale)
        and phase_scale <= _MPS_MAX_ECCENTRIC_PHASE
    )


def _taylorf2ecc_samples(
    inputs,
    frequencies,
    *,
    f_ecc,
    eccentricity,
    eccentricity_order,
    time_shift=0.0,
):
    """Evaluate TaylorF2 samples and apply the eccentric phase on-device."""
    import torch

    samples = _taylorf2_samples(inputs, frequencies, time_shift=time_shift)
    if eccentricity == 0.0:
        return samples

    total_mass = inputs.mass1 + inputs.mass2
    eta = inputs.mass1 * inputs.mass2 / total_mass**2
    pi_mass = math.pi * total_mass * lal.MTSUN_SI
    velocity = torch.pow(pi_mass * frequencies, 1.0 / 3.0)
    velocity0 = torch.pow(
        torch.as_tensor(
            pi_mass * f_ecc,
            dtype=inputs.real_dtype,
            device=inputs.device,
        ),
        1.0 / 3.0,
    )
    eccentric_phase = _eccentric_phase_polynomial(
        velocity,
        velocity0,
        eccentricity,
        eta,
        eccentricity_order,
    ) / velocity**5

    if inputs.f_ref == 0.0:
        reference_phase = torch.zeros(
            (), dtype=inputs.real_dtype, device=inputs.device
        )
    else:
        reference_velocity = torch.pow(
            torch.as_tensor(
                pi_mass * inputs.f_ref,
                dtype=inputs.real_dtype,
                device=inputs.device,
            ),
            1.0 / 3.0,
        )
        reference_phase = _eccentric_phase_polynomial(
            reference_velocity,
            velocity0,
            eccentricity,
            eta,
            eccentricity_order,
        ) / reference_velocity**5

    delta_phase = eccentric_phase - reference_phase
    rotation = torch.complex(torch.cos(delta_phase), -torch.sin(delta_phase))
    return samples * rotation.to(inputs.complex_dtype)


def taylorf2ecc_fd_sequence_torch(**params):
    """Evaluate TaylorF2Ecc at arbitrary frequencies with Torch.

    This is a native extension because LAL has no TaylorF2Ecc sequence
    implementation. When ``f_ref`` is zero, the lowest sample point is used
    as the eccentricity reference, mirroring the regular-grid ``f_lower``
    convention without moving the scalar off-device.
    """
    import torch

    from pycbc.types import Array as PyCBCArray
    from pycbc.types.array_torch import TorchArrayData

    if not taylorf2ecc_sequence_native_supported(params):
        raise ValueError(
            "TaylorF2Ecc sequence parameters are not supported by the native "
            "Torch path"
        )

    circular_params = dict(params)
    circular_params["eccentricity"] = 0.0
    circular_params["eccentricity_order"] = -1
    inputs = _taylorf2_inputs(circular_params, sequence=True)
    frequencies = _taylorf2ecc_sequence_frequencies(
        params["sample_points"],
        inputs,
    )
    f_ecc = inputs.f_ref if inputs.f_ref != 0.0 else torch.min(frequencies)
    samples = _taylorf2ecc_samples(
        inputs,
        frequencies,
        f_ecc=f_ecc,
        eccentricity=float(params.get("eccentricity", 0.0)),
        eccentricity_order=_as_order(params.get("eccentricity_order", -1)),
    )
    plus, cross = _taylorf2_polarizations(samples, inputs)
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


def taylorf2ecc_fd_torch(**params):
    """Generate TaylorF2Ecc polarizations on the active Torch device."""
    import torch

    from pycbc.types import FrequencySeries
    from pycbc.types.array_torch import TorchArrayData

    if not taylorf2ecc_native_supported(params):
        raise ValueError(
            "TaylorF2Ecc parameters are not supported by the native Torch path"
        )

    circular_params = dict(params)
    circular_params["eccentricity"] = 0.0
    circular_params["eccentricity_order"] = -1
    inputs = _taylorf2_inputs(circular_params)

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    eccentricity = float(params.get("eccentricity", 0.0))
    eccentricity_order = _as_order(params.get("eccentricity_order", -1))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("TaylorF2Ecc frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("TaylorF2Ecc delta_f and f_lower must be positive")

    pi_mass = math.pi * (inputs.mass1 + inputs.mass2) * lal.MTSUN_SI
    f_isco = 1.0 / (6.0**1.5 * pi_mass)
    if f_final == 0.0:
        if inputs.tidal_order == 0:
            f_max = f_isco
        else:
            f_max = min(
                f_isco,
                _contact_frequency(
                    inputs.mass1,
                    inputs.mass2,
                    inputs.lambda1,
                    inputs.lambda2,
                ),
            )
    else:
        f_max = f_final
    if f_max <= f_lower:
        raise ValueError("TaylorF2Ecc ending frequency must exceed f_lower")

    length = int(f_max / delta_f + 1.0)
    first_bin = int(math.ceil(f_lower / delta_f))
    if first_bin >= length:
        raise ValueError("TaylorF2Ecc frequency range contains no sampled bins")

    raw = torch.zeros(
        length,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    frequencies = (
        torch.arange(
            first_bin,
            length,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        * delta_f
    )
    epoch = -1.0 / delta_f
    # LAL defaults the eccentricity reference to f_ref, or f_lower when
    # f_ref=0. PyCBC does not expose LAL's separate f_ecc dictionary entry.
    f_ecc = inputs.f_ref if inputs.f_ref != 0.0 else f_lower
    raw[first_bin:] = _taylorf2ecc_samples(
        inputs,
        frequencies,
        f_ecc=f_ecc,
        eccentricity=eccentricity,
        eccentricity_order=eccentricity_order,
        time_shift=epoch,
    )
    plus, cross = _taylorf2_polarizations(raw, inputs)
    return (
        FrequencySeries(
            TorchArrayData(plus), delta_f=delta_f, epoch=epoch, copy=False
        ),
        FrequencySeries(
            TorchArrayData(cross), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )
