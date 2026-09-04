# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch backend for time- and frequency-domain ringdown waveforms."""

import numpy

import pycbc.scheme as _scheme
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.types.backend import backend_array
from pycbc.waveform import ringdown as _common


def _device_and_dtype():
    """Return the active Torch device and its preferred real dtype."""
    import torch

    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("Torch ringdown requires an active TorchScheme")
    dtype = torch.float32 if state.torch_device.type == 'mps' \
        else torch.float64
    return state.torch_device, dtype


def _vector(values):
    """Move an evaluation grid to the active Torch device."""
    import torch

    device, dtype = _device_and_dtype()
    native = backend_array(values, "torch")
    if native is not None:
        values = native
    return torch.as_tensor(values, dtype=dtype, device=device)


def _zeros(length, complex_output=False):
    """Create ringdown storage on the active Torch device."""
    import torch

    device, dtype = _device_and_dtype()
    if complex_output:
        dtype = torch.complex64 if dtype == torch.float32 \
            else torch.complex128
    return TorchArrayData(torch.zeros(length, dtype=dtype, device=device))


def td_output_vector(freqs, damping_times, taper=False,
                     delta_t=None, t_final=None):
    """Create empty time-domain ringdown series on the Torch device."""
    if not delta_t:
        delta_t = _common.lm_deltat(freqs, damping_times)
    if not t_final:
        t_final = _common.lm_tfinal(damping_times)
    kmax = int(t_final / delta_t) + 1
    if taper:
        max_tau = max(damping_times.values()) if \
            isinstance(damping_times, dict) else damping_times
        kmax += int(max_tau / delta_t)
    outplus = TimeSeries(_zeros(kmax), delta_t=delta_t, copy=False)
    outcross = TimeSeries(_zeros(kmax), delta_t=delta_t, copy=False)
    if taper:
        start = -max_tau
        start -= start % delta_t
        outplus._epoch, outcross._epoch = start, start
    return outplus, outcross


def fd_output_vector(freqs, damping_times, delta_f=None, f_final=None):
    """Create empty frequency-domain ringdown series on the Torch device."""
    if not delta_f:
        delta_f = _common.lm_deltaf(damping_times)
    if not f_final:
        f_final = _common.lm_ffinal(freqs, damping_times)
    kmax = int(f_final / delta_f) + 1
    outplus = FrequencySeries(
        _zeros(kmax, complex_output=True), delta_f=delta_f, copy=False)
    outcross = FrequencySeries(
        _zeros(kmax, complex_output=True), delta_f=delta_f, copy=False)
    return outplus, outcross


def _spher_harms(grid, **kwargs):
    """Evaluate harmonics on the same device as a Torch waveform grid."""
    if kwargs.get('harmonics', 'spherical') != 'spherical':
        return _common.spher_harms(**kwargs)

    from pycbc.waveform._spherical_harmonics_torch import (
        spin_weighted_spherical_harmonic,
    )

    common = dict(
        theta=kwargs.get('inclination', 0.),
        phi=kwargs.get('azimuthal', 0.),
        spin_weight=-2,
        ell=kwargs['l'],
        dtype=grid.dtype,
        device=grid.device,
    )
    return (
        spin_weighted_spherical_harmonic(emm=kwargs['m'], **common),
        spin_weighted_spherical_harmonic(emm=-kwargs['m'], **common),
    )


def td_damped_sinusoid(f_0, tau, amp, phi, times,
                       l=2, m=2, n=0, inclination=0., azimuthal=0.,
                       dphi=0., dbeta=0., harmonics='spherical',
                       final_spin=None, pol=None, polnm=None):
    """Evaluate a time-domain damped sinusoid with Torch."""
    import torch

    times = _vector(times)
    xlm, xlnm = _spher_harms(
        times, harmonics=harmonics, l=l, m=m, n=n,
        inclination=inclination, azimuthal=azimuthal,
        spin=final_spin, pol=pol, polnm=polnm)

    omegalm = _common.two_pi * f_0 * times
    damping = torch.where(times < 0, 10 * times / tau, -times / tau)
    if m == 0:
        hlm = xlm * amp * torch.exp(damping + 1j * (omegalm + phi))
    else:
        if dbeta == 0:
            alm = alnm = amp
        else:
            beta = _common.pi / 4 + dbeta
            alm = 2**0.5 * amp * numpy.cos(beta)
            alnm = 2**0.5 * amp * numpy.sin(beta)
        phinm = l * _common.pi + dphi - phi
        hlm = (
            xlm * alm * torch.exp(damping + 1j * (omegalm + phi))
            + xlnm * alnm
            * torch.exp(damping - 1j * (omegalm - phinm))
        )
    return hlm.real, hlm.imag


def fd_damped_sinusoid(f_0, tau, amp, phi, freqs, t_0=0.,
                       l=2, m=2, n=0, inclination=0., azimuthal=0.,
                       harmonics='spherical', final_spin=None,
                       pol=None, polnm=None):
    """Evaluate a frequency-domain damped sinusoid with Torch."""
    import torch

    freqs = _vector(freqs)
    if inclination is None:
        inclination = 0.
    if azimuthal is None:
        azimuthal = 0.
    xlm, xlnm = _spher_harms(
        freqs, harmonics=harmonics, l=l, m=m, n=n,
        inclination=inclination, azimuthal=azimuthal,
        spin=final_spin, pol=pol, polnm=polnm)
    xp = xlm + (-1)**l * xlnm
    xc = xlm - (-1)**l * xlnm
    denominator = (
        1 + (4j * _common.pi * freqs * tau)
        - 4 * _common.pi_sq * (freqs * freqs - f_0 * f_0) * tau * tau
    )
    norm = amp * tau / denominator
    if t_0 != 0:
        norm *= torch.exp(-1j * _common.two_pi * freqs * t_0)
    a1 = 1 + 2j * _common.pi * freqs * tau
    a2 = _common.two_pi * f_0 * tau
    hptilde = norm * xp * (
        a1 * numpy.cos(phi) - a2 * numpy.sin(phi))
    hctilde = norm * xc * (
        a1 * numpy.sin(phi) + a2 * numpy.cos(phi))
    return hptilde, hctilde


def multimode_base(input_params, domain, freq_tau_approximant=False):
    """Generate a multimode ringdown entirely on the Torch device."""
    import torch

    input_params['lmns'] = _common.format_lmns(input_params['lmns'])
    amps, phis, dbetas, dphis = _common.lm_amps_phases(**input_params)
    pols, polnms = _common.lm_arbitrary_harmonics(**input_params)
    harmonics = input_params.get('harmonics', 'spherical')
    final_spin = input_params['final_spin'] \
        if harmonics == 'spheroidal' else None
    input_params.setdefault('inclination', 0.)
    input_params.setdefault('azimuthal', 0.)

    if freq_tau_approximant:
        freqs, taus = _common.lm_freqs_taus(**input_params)
        norm = 1.
    else:
        freqs, taus = _common.get_lm_f0tau_allmodes(
            input_params['final_mass'], input_params['final_spin'],
            input_params['lmns'])
        norm = _common.Kerr_factor(
            input_params['final_mass'], input_params['distance']) \
            if 'distance' in input_params else 1.
        for mode, freq in freqs.items():
            if 'delta_f{}'.format(mode) in input_params:
                freqs[mode] += input_params["delta_f{}".format(mode)] * freq
        for mode, tau in taus.items():
            if 'delta_tau{}'.format(mode) in input_params:
                taus[mode] += input_params["delta_tau{}".format(mode)] * tau

    device, dtype = _device_and_dtype()
    if domain == 'td':
        outplus, outcross = td_output_vector(
            freqs, taus, input_params['taper'], input_params['delta_t'],
            input_params['t_final'])
        sample_grid = (
            torch.arange(len(outplus), device=device, dtype=dtype)
            * outplus.delta_t + float(outplus.start_time)
        )
        start = None
    elif domain == 'fd':
        outplus, outcross = fd_output_vector(
            freqs, taus, input_params['delta_f'], input_params['f_final'])
        start = int((input_params['f_lower'] or 0.) / outplus.delta_f)
        sample_grid = (
            torch.arange(start, len(outplus), device=device, dtype=dtype)
            * outplus.delta_f
        )
    else:
        raise ValueError(
            'unrecognised domain argument {}; must be either fd or td'.format(
                domain))

    for lmn in freqs:
        if amps[lmn] == 0.:
            continue
        common = dict(
            l=int(lmn[0]), m=int(lmn[1]), n=int(lmn[2]),
            inclination=input_params['inclination'],
            azimuthal=input_params['azimuthal'], harmonics=harmonics,
            final_spin=final_spin, pol=pols[lmn], polnm=polnms[lmn])
        if domain == 'td':
            hplus, hcross = td_damped_sinusoid(
                freqs[lmn], taus[lmn], amps[lmn], phis[lmn], sample_grid,
                dphi=dphis[lmn], dbeta=dbetas[lmn], **common)
            backend_array(outplus, "torch").add_(hplus)
            backend_array(outcross, "torch").add_(hcross)
        else:
            hplus, hcross = fd_damped_sinusoid(
                freqs[lmn], taus[lmn], amps[lmn], phis[lmn], sample_grid,
                t_0=input_params['t_0'], **common)
            backend_array(outplus, "torch")[start:].add_(hplus)
            backend_array(outcross, "torch")[start:].add_(hcross)
    return norm * outplus, norm * outcross


__all__ = (
    'fd_damped_sinusoid',
    'fd_output_vector',
    'multimode_base',
    'td_damped_sinusoid',
    'td_output_vector',
)
