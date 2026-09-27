"""JAX implementation of sine-Gaussian chisq values."""

import numpy
from pycbc.waveform.utils import apply_fseries_time_shift
from pycbc.filter import sigma
from pycbc.events import ranking

def values(calculator, stilde, template, psd, snrv, snr_norm, bchisq, bchisq_dof, indices):
    import jax.numpy as jnp
    from pycbc.waveform import sinegauss_jax
    if template.params.template_hash not in calculator.params:
        return jnp.ones(len(snrv))
    values = calculator.params[template.params.template_hash].split(',')

    # Get the chisq bins to use as the frequency reference point
    bins = calculator.cached_chisq_bins(template, psd)

    # This is implemented slowly, so let's not call it often, OK?
    chisq = jnp.ones(len(snrv))
    gtem = [None for _ in values]
    for i, snrvi in enumerate(snrv):
        #Skip if newsnr too low
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

            #Calculate sine-gaussian tile
            if gtem[idxx] is None:
                # These are always the same values for a template, so
                # if computing 10 sgchisq points, don't want to call
                # this 10 times (for each SG template)
                gtem[idxx] = sinegauss_jax.fd_sine_gaussian(
                    1.0, q, fcen, flow,
                    len(template) * template.delta_f,
                    template.delta_f).astype(numpy.complex64)
            gsigma = sigma(gtem[idxx], psd=psd,
                                 low_frequency_cutoff=flow,
                                 high_frequency_cutoff=fhigh)
            #Calculate the SNR of the tile
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
