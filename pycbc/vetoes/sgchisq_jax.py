"""JAX implementation of sine-Gaussian chisq values."""

import math

import jax
import jax.numpy as jnp
import numpy

from pycbc.types import FrequencySeries
from pycbc.types.array_jax import JAXArrayData, _reference_enabled, to_jax
from pycbc.waveform.utils import apply_fseries_time_shift
from pycbc.filter import sigma
from pycbc.events import ranking


def values(
    calculator,
    stilde,
    template,
    psd,
    snrv,
    snr_norm,
    bchisq,
    bchisq_dof,
    indices,
):
    if template.params.template_hash not in calculator.params:
        return jnp.ones(len(snrv))
    if _reference_enabled("sgchisq"):
        from pycbc.reference_jax import cpu_reference

        result = cpu_reference(
            "sgchisq",
            jax.device_get(to_jax(stilde)),
            stilde.delta_f,
            stilde.epoch,
            template=numpy.asarray(jax.device_get(to_jax(template))),
            template_epoch=(
                None if template.epoch is None else str(template.epoch)
            ),
            f_lower=template.f_lower,
            template_hash=template.params.template_hash,
            params=calculator.params,
            snr_threshold=calculator.snr_threshold,
            bins=numpy.asarray(calculator.cached_chisq_bins(template, psd)),
            psd=numpy.asarray(jax.device_get(to_jax(psd))),
            snrv=numpy.asarray(snrv),
            snr_norm=snr_norm,
            bchisq=numpy.asarray(bchisq),
            bchisq_dof=numpy.asarray(bchisq_dof),
            indices=numpy.asarray(indices),
        )
        return jax.device_put(result, to_jax(stilde).device)
    values = calculator.params[template.params.template_hash].split(',')

    # Get the chisq bins to use as the frequency reference point
    bins = calculator.cached_chisq_bins(template, psd)

    # This is implemented slowly, so let's not call it often, OK?
    chisq = jnp.ones(len(snrv))
    gtem = [None for _ in values]
    for i, snrvi in enumerate(snrv):
        # Skip if newsnr too low
        snr = abs(snrvi * snr_norm)
        nsnr = ranking.newsnr(snr, bchisq[i] / bchisq_dof[i])
        if nsnr < calculator.snr_threshold:
            continue

        N = (len(template) - 1) * 2
        dt = 1.0 / (N * template.delta_f)
        kmin = int(template.f_lower / psd.delta_f)
        time = float(template.epoch) + dt * indices[i]
        # Shift the time of interest to be centered on 0
        stilde_shift = apply_fseries_time_shift(stilde, -time)

        # Only apply the sine-Gaussian in a +-50 Hz range around the
        # central frequency
        qwindow = 50
        chisq = chisq.at[i].set(0)

        # Estimate the maximum frequency up to which the waveform has
        # power by approximating power per frequency
        # as constant over the last 2 chisq bins. We cannot use the final
        # chisq bin edge as it does not have to be where the waveform
        # terminates.
        fstep = (bins[-2] - bins[-3])
        fpeak = (bins[-2] + fstep) * template.delta_f

        # This is 90% of the Nyquist frequency of the data
        # This allows us to avoid issues near Nyquist due to resample
        # Filtering
        fstop = len(stilde) * stilde.delta_f * 0.9

        dof = 0
        # Calculate the sum of SNR^2 for the sine-Gaussians specified
        for idxx, descr in enumerate(values):
            # Get the q and frequency offset from the descriptor
            q, offset = descr.split('-')
            q, offset = float(q), float(offset)
            fcen = fpeak + offset
            flow = max(kmin * template.delta_f, fcen - qwindow)
            fhigh = fcen + qwindow

            # If any sine-gaussian tile has an upper frequency near
            # nyquist return 1 instead.
            if fhigh > fstop:
                return jnp.ones(len(snrv))

            kmin = int(flow / template.delta_f)
            kmax = int(fhigh / template.delta_f)

            # Calculate sine-gaussian tile
            if gtem[idxx] is None:
                # These are always the same values for a template, so
                # if computing 10 sgchisq points, don't want to call
                # this 10 times (for each SG template)
                gtem[idxx] = _sine_gaussian_basis(
                    1.0, q, fcen, flow,
                    len(template) * template.delta_f,
                    template.delta_f).astype(numpy.complex64)
            gsigma = sigma(
                gtem[idxx],
                psd=psd,
                low_frequency_cutoff=flow,
                high_frequency_cutoff=fhigh,
            )
            # Calculate the SNR of the tile
            gsnr = (gtem[idxx][kmin:kmax] * stilde_shift[kmin:kmax]).sum()
            gsnr *= 4.0 * gtem[idxx].delta_f / gsigma
            increment = abs(gsnr)**2.0
            chisq = chisq.at[i].add(increment)
            dof += 2
        if dof == 0:
            chisq = chisq.at[i].set(1)
        else:
            chisq = chisq.at[i].set(chisq[i] / dof)
    return chisq


def _sine_gaussian_basis(amp, quality, central_frequency, fmin, fmax, delta_f):
    """Construct the private on-device tile used by the SG chi-square veto."""
    if _reference_enabled("sg_basis"):
        from pycbc.reference_jax import cpu_reference

        values, spacing, epoch = cpu_reference(
            "sg_basis",
            amp=amp,
            quality=quality,
            central_frequency=central_frequency,
            fmin=fmin,
            fmax=fmax,
            delta_f=delta_f,
        )
        return FrequencySeries(values, delta_f=spacing, epoch=epoch)
    kmin = int(round(fmin / delta_f))
    kmax = int(round(fmax / delta_f))
    pi = math.pi
    tau = quality / (2 * pi * central_frequency)
    f = jnp.arange(kmax, dtype=jnp.float64) * delta_f
    out = jnp.zeros((kmax,), dtype=jnp.complex128)
    cutoff = -50.0
    low = max(kmin, int((central_frequency - (-cutoff)**.5 /
                         (tau * pi)) // delta_f))
    high = min(kmax, int((central_frequency + (-cutoff)**.5 /
                          (tau * pi)) // delta_f))
    term = -(tau * pi * (f[low:high] - central_frequency)) ** 2
    aterm = amp * pi**.5 / 2 * tau
    out = out.at[low:high].set(aterm * jnp.exp(term))
    high2 = int(-cutoff / quality**2 * central_frequency // delta_f)
    if high2 > kmin:
        term2 = -quality**2 * f[kmin:high2] / central_frequency
        out = out.at[kmin:high2].multiply(1 + jnp.exp(term2))
    return FrequencySeries(JAXArrayData(out), delta_f=delta_f, copy=False)
