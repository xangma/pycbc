# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""JAX backend for time- and frequency-domain ringdown waveforms."""

import numpy as np
import jax.numpy as jnp

from pycbc.types import FrequencySeries, TimeSeries
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, to_jax
from pycbc.waveform import ringdown as _common
from pycbc.waveform._spherical_harmonics_jax import spin_weighted_spherical_harmonic


def _vector(values):
    """Move an evaluation grid to JAX array in float64."""
    _ensure_x64()
    return to_jax(values).astype(jnp.float64)


def _zeros(length, complex_output=False):
    """Create ringdown storage in JAX."""
    _ensure_x64()
    dtype = jnp.complex128 if complex_output else jnp.float64
    return JAXArrayData(jnp.zeros(length, dtype=dtype))


def td_output_vector(freqs, damping_times, taper=False, delta_t=None, t_final=None):
    """Create empty time-domain ringdown series on JAX backend."""
    if not delta_t:
        delta_t = _common.lm_deltat(freqs, damping_times)
    if not t_final:
        t_final = _common.lm_tfinal(damping_times)
    kmax = int(t_final / delta_t) + 1
    if taper:
        max_tau = (
            max(damping_times.values())
            if isinstance(damping_times, dict)
            else damping_times
        )
        kmax += int(max_tau / delta_t)
    outplus = TimeSeries(_zeros(kmax), delta_t=delta_t, copy=False)
    outcross = TimeSeries(_zeros(kmax), delta_t=delta_t, copy=False)
    if taper:
        start = -max_tau
        start -= start % delta_t
        outplus._epoch, outcross._epoch = start, start
    return outplus, outcross


def fd_output_vector(freqs, damping_times, delta_f=None, f_final=None):
    """Create empty frequency-domain ringdown series on JAX backend."""
    if not delta_f:
        delta_f = _common.lm_deltaf(damping_times)
    if not f_final:
        f_final = _common.lm_ffinal(freqs, damping_times)
    kmax = int(f_final / delta_f) + 1
    outplus = FrequencySeries(
        _zeros(kmax, complex_output=True), delta_f=delta_f, copy=False
    )
    outcross = FrequencySeries(
        _zeros(kmax, complex_output=True), delta_f=delta_f, copy=False
    )
    return outplus, outcross


def _spher_harms(grid, **kwargs):
    """Evaluate harmonics for JAX waveform grid."""
    if kwargs.get("harmonics", "spherical") != "spherical":
        return _common.spher_harms(**kwargs)

    theta = kwargs.get("inclination", 0.0)
    phi = kwargs.get("azimuthal", 0.0)
    ell = kwargs["l"]
    emm = kwargs["m"]

    xlm = spin_weighted_spherical_harmonic(theta, phi, -2, ell, emm, dtype=jnp.float64)
    xlnm = spin_weighted_spherical_harmonic(theta, phi, -2, ell, -emm, dtype=jnp.float64)
    return xlm, xlnm


def td_damped_sinusoid(
    f_0,
    tau,
    amp,
    phi,
    times,
    l=2,
    m=2,
    n=0,
    inclination=0.0,
    azimuthal=0.0,
    dphi=0.0,
    dbeta=0.0,
    harmonics="spherical",
    final_spin=None,
    pol=None,
    polnm=None,
):
    """Evaluate a time-domain damped sinusoid with JAX."""
    _ensure_x64()
    times = _vector(times)
    xlm, xlnm = _spher_harms(
        times,
        harmonics=harmonics,
        l=l,
        m=m,
        n=n,
        inclination=inclination,
        azimuthal=azimuthal,
        spin=final_spin,
        pol=pol,
        polnm=polnm,
    )

    omegalm = _common.two_pi * f_0 * times
    damping = jnp.where(times < 0, 10.0 * times / tau, -times / tau)
    decay = jnp.exp(damping)
    phase1 = omegalm + phi
    cos1 = jnp.cos(phase1)
    sin1 = jnp.sin(phase1)
    xr, xi = xlm.real, xlm.imag

    if m == 0:
        scale = amp * decay
        re = scale * (xr * cos1 - xi * sin1)
        im = scale * (xr * sin1 + xi * cos1)
        return re, im

    if dbeta == 0:
        alm = alnm = amp
    else:
        beta = _common.pi / 4.0 + dbeta
        alm = (2.0 ** 0.5) * amp * np.cos(beta)
        alnm = (2.0 ** 0.5) * amp * np.sin(beta)
    phinm = l * _common.pi + dphi - phi
    phase2 = omegalm - phinm
    cos2 = jnp.cos(phase2)
    sin2 = jnp.sin(phase2)
    yr, yi = xlnm.real, xlnm.imag

    z1_re = alm * decay * (xr * cos1 - xi * sin1)
    z1_im = alm * decay * (xr * sin1 + xi * cos1)
    z2_re = alnm * decay * (yr * cos2 + yi * sin2)
    z2_im = alnm * decay * (yi * cos2 - yr * sin2)

    return z1_re + z2_re, z1_im + z2_im


def fd_damped_sinusoid(
    f_0,
    tau,
    amp,
    phi,
    freqs,
    t_0=0.0,
    l=2,
    m=2,
    n=0,
    inclination=0.0,
    azimuthal=0.0,
    harmonics="spherical",
    final_spin=None,
    pol=None,
    polnm=None,
):
    """Evaluate a frequency-domain damped sinusoid with JAX."""
    _ensure_x64()
    freqs = _vector(freqs)
    if inclination is None:
        inclination = 0.0
    if azimuthal is None:
        azimuthal = 0.0
    xlm, xlnm = _spher_harms(
        freqs,
        harmonics=harmonics,
        l=l,
        m=m,
        n=n,
        inclination=inclination,
        azimuthal=azimuthal,
        spin=final_spin,
        pol=pol,
        polnm=polnm,
    )
    xr, xi = xlm.real, xlm.imag
    yr, yi = xlnm.real, xlnm.imag
    sgn = (-1) ** l
    xp_r = xr + sgn * yr
    xp_i = xi + sgn * yi
    xc_r = xr - sgn * yr
    xc_i = xi - sgn * yi

    denom_r = 1.0 - 4.0 * _common.pi_sq * (freqs * freqs - f_0 * f_0) * (tau * tau)
    denom_i = 4.0 * _common.pi * freqs * tau
    denom_mag_sq = denom_r * denom_r + denom_i * denom_i

    scale = (amp * tau) / denom_mag_sq
    norm_r = scale * denom_r
    norm_i = -scale * denom_i

    if t_0 != 0:
        shift_phase = -_common.two_pi * freqs * t_0
        shift_r = jnp.cos(shift_phase)
        shift_i = jnp.sin(shift_phase)
        n_r = norm_r * shift_r - norm_i * shift_i
        n_i = norm_r * shift_i + norm_i * shift_r
        norm_r, norm_i = n_r, n_i

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    two_pi_tau = _common.two_pi * tau
    two_pi_f_tau = two_pi_tau * freqs
    a2 = _common.two_pi * f_0 * tau

    bp_r = cos_phi - a2 * sin_phi
    bp_i = two_pi_f_tau * cos_phi

    bc_r = sin_phi + a2 * cos_phi
    bc_i = two_pi_f_tau * sin_phi

    tp_r = norm_r * xp_r - norm_i * xp_i
    tp_i = norm_r * xp_i + norm_i * xp_r
    hp_r = tp_r * bp_r - tp_i * bp_i
    hp_i = tp_r * bp_i + tp_i * bp_r

    tc_r = norm_r * xc_r - norm_i * xc_i
    tc_i = norm_r * xc_i + norm_i * xc_r
    hc_r = tc_r * bc_r - tc_i * bc_i
    hc_i = tc_r * bc_i + tc_i * bc_r

    return (hp_r + 1j * hp_i).astype(jnp.complex128), (hc_r + 1j * hc_i).astype(jnp.complex128)


def multimode_base(input_params, domain, freq_tau_approximant=False):
    """Generate a multimode ringdown entirely on the JAX device."""
    _ensure_x64()
    input_params["lmns"] = _common.format_lmns(input_params["lmns"])
    amps, phis, dbetas, dphis = _common.lm_amps_phases(**input_params)
    pols, polnms = _common.lm_arbitrary_harmonics(**input_params)
    harmonics = input_params.get("harmonics", "spherical")
    final_spin = input_params["final_spin"] if harmonics == "spheroidal" else None
    input_params.setdefault("inclination", 0.0)
    input_params.setdefault("azimuthal", 0.0)

    if freq_tau_approximant:
        freqs, taus = _common.lm_freqs_taus(**input_params)
        norm = 1.0
    else:
        freqs, taus = _common.get_lm_f0tau_allmodes(
            input_params["final_mass"], input_params["final_spin"], input_params["lmns"]
        )
        norm = (
            _common.Kerr_factor(input_params["final_mass"], input_params["distance"])
            if "distance" in input_params
            else 1.0
        )
        for mode, freq in freqs.items():
            if "delta_f{}".format(mode) in input_params:
                freqs[mode] += input_params["delta_f{}".format(mode)] * freq
        for mode, tau in taus.items():
            if "delta_tau{}".format(mode) in input_params:
                taus[mode] += input_params["delta_tau{}".format(mode)] * tau

    if domain == "td":
        outplus, outcross = td_output_vector(
            freqs,
            taus,
            input_params["taper"],
            input_params["delta_t"],
            input_params["t_final"],
        )
        sample_grid = (
            jnp.arange(len(outplus), dtype=jnp.float64) * outplus.delta_t
            + float(outplus.start_time)
        )
        start = None
    elif domain == "fd":
        outplus, outcross = fd_output_vector(
            freqs, taus, input_params["delta_f"], input_params["f_final"]
        )
        start = int((input_params["f_lower"] or 0.0) / outplus.delta_f)
        sample_grid = (
            jnp.arange(start, len(outplus), dtype=jnp.float64) * outplus.delta_f
        )
    else:
        raise ValueError("Invalid domain: must be 'td' or 'fd'")

    for lmn in freqs:
        if amps[lmn] == 0.0:
            continue
        if domain == "td":
            hplus, hcross = td_damped_sinusoid(
                freqs[lmn],
                taus[lmn],
                amps[lmn],
                phis[lmn],
                sample_grid,
                l=int(lmn[0]),
                m=int(lmn[1]),
                n=int(lmn[2]),
                inclination=input_params["inclination"],
                azimuthal=input_params["azimuthal"],
                dphi=dphis[lmn],
                dbeta=dbetas[lmn],
                harmonics=harmonics,
                final_spin=final_spin,
                pol=pols[lmn],
                polnm=polnms[lmn],
            )
            outplus._data.set_array(outplus._data.array + hplus)
            outcross._data.set_array(outcross._data.array + hcross)
        elif domain == "fd":
            hplus, hcross = fd_damped_sinusoid(
                freqs[lmn],
                taus[lmn],
                amps[lmn],
                phis[lmn],
                sample_grid,
                t_0=input_params.get("t_0", 0.0),
                l=int(lmn[0]),
                m=int(lmn[1]),
                n=int(lmn[2]),
                inclination=input_params["inclination"],
                azimuthal=input_params["azimuthal"],
                harmonics=harmonics,
                final_spin=final_spin,
                pol=pols[lmn],
                polnm=polnms[lmn],
            )
            outplus._data.set_slice(slice(start, start + len(hplus)), outplus._data.array[start:start + len(hplus)] + hplus)
            outcross._data.set_slice(slice(start, start + len(hcross)), outcross._data.array[start:start + len(hcross)] + hcross)

    return norm * outplus, norm * outcross
