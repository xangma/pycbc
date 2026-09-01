"""Chisq based on sine-gaussian tiles.
See https://arxiv.org/abs/1709.08974 for a discussion.
"""

from functools import lru_cache
import numpy

from pycbc.waveform.utils import apply_fseries_time_shift
from pycbc.filter import sigma
from pycbc.waveform import sinegauss
from pycbc.vetoes.chisq import (
    SingleDetPowerChisq,
    _as_torch_tensor,
    _torch_array,
    _torch_tensor,
)
from pycbc.events import ranking


@lru_cache(maxsize=512)
def _cached_gpu_sg_tile(
    device_type,
    device_index,
    complex_dtype,
    template_len,
    template_delta_f,
    kmin_template,
    fpeak,
    descr,
):
    """Precompute and cache GPU Sine-Gaussian tile frequency-domain slice."""
    import torch

    quality, offset = (float(value) for value in descr.split("-"))
    central_frequency = fpeak + offset
    qwindow = 50.0
    flow = max(kmin_template * template_delta_f, central_frequency - qwindow)
    fhigh = central_frequency + qwindow
    kmin = int(flow / template_delta_f)
    kmax = int(fhigh / template_delta_f)
    total_duration = template_len * template_delta_f
    tile = sinegauss.fd_sine_gaussian(
        1.0,
        quality,
        central_frequency,
        flow,
        total_duration,
        template_delta_f,
    ).astype(numpy.complex64)
    device = (
        torch.device(device_type, device_index)
        if device_index is not None
        else torch.device(device_type)
    )
    tile_slice = torch.as_tensor(
        tile[kmin:kmax], device=device, dtype=complex_dtype
    )
    return tile_slice, kmin, kmax, fhigh


def _torch_ones(reference, length):
    """Allocate SG chi-squared defaults on the active Torch device."""
    import torch

    return _torch_array(
        torch.ones(length, device=reference.device, dtype=torch.float32)
    )


def _torch_sgchisq_values(
    stilde,
    template,
    psd,
    snrv,
    snr_norm,
    bchisq,
    bchisq_dof,
    indices,
    values,
    bins,
    threshold,
):
    """Evaluate all sine-Gaussian tiles without reducing through the host."""
    import torch

    stilde_tensor = _torch_tensor(stilde)
    psd_tensor = _torch_tensor(psd)
    length = len(snrv)
    output = torch.ones(
        length, device=stilde_tensor.device, dtype=torch.float32
    )

    # MPS does not support the double-precision accumulation used elsewhere
    # by PyCBC. Accumulate at the highest precision available on each device.
    if stilde_tensor.device.type == "mps":
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    else:
        real_dtype = torch.float64
        complex_dtype = torch.complex128

    snr_values = _as_torch_tensor(
        snrv, stilde_tensor, dtype=complex_dtype
    )
    bchisq_values = _as_torch_tensor(
        bchisq, stilde_tensor, dtype=real_dtype
    )
    dof_values = _as_torch_tensor(
        bchisq_dof, stilde_tensor, dtype=real_dtype
    )
    index_values = _as_torch_tensor(
        indices, stilde_tensor, dtype=torch.long
    )
    snr = torch.abs(snr_values * float(snr_norm))
    reduced_chisq = bchisq_values / dof_values
    newsnr = torch.where(
        reduced_chisq > 1,
        snr * (0.5 * (1 + reduced_chisq**3)) ** (-1.0 / 6.0),
        snr,
    )
    # A NaN newsnr follows the existing scalar path and is evaluated rather
    # than skipped because ``nan < threshold`` is false.
    active = ~(newsnr < threshold)

    sample_count = (len(template) - 1) * 2
    delta_t = 1.0 / (sample_count * template.delta_f)
    times = float(template.epoch) + delta_t * index_values.to(real_dtype)
    kmin_template = int(template.f_lower / psd.delta_f)
    fstep = bins[-2] - bins[-3]
    fpeak = (bins[-2] + fstep) * template.delta_f
    fstop = len(stilde) * stilde.delta_f * 0.9
    chisq = torch.zeros(length, device=stilde_tensor.device, dtype=real_dtype)
    dof = 0

    for descr in values:
        tile_tensor, kmin, kmax, fhigh = _cached_gpu_sg_tile(
            stilde_tensor.device.type,
            stilde_tensor.device.index,
            complex_dtype,
            len(template),
            template.delta_f,
            kmin_template,
            fpeak,
            descr,
        )
        if fhigh > fstop:
            return _torch_array(output)

        psd_slice = psd_tensor[kmin:kmax].to(real_dtype)
        tile_power = torch.sum(
            torch.abs(tile_tensor) ** 2 / psd_slice,
            dtype=real_dtype,
        )
        tile_sigma = torch.sqrt(4.0 * template.delta_f * tile_power)

        frequencies = torch.arange(
            kmin,
            kmax,
            device=stilde_tensor.device,
            dtype=stilde_tensor.real.dtype,
        )
        # Match the existing single-precision time-shift phase when the
        # overwhitened strain is complex64, while accumulating correlations
        # in double precision where the device supports it.
        phase_step = (
            2.0 * torch.pi * times * stilde.delta_f
        ).to(stilde_tensor.real.dtype)
        phase = torch.exp(
            1j * phase_step[:, None] * frequencies[None, :]
        ).to(complex_dtype)
        base = (
            tile_tensor
            * stilde_tensor[kmin:kmax].to(complex_dtype)
        )
        tile_snr = torch.sum(
            phase * base[None, :], dim=1, dtype=complex_dtype
        )
        tile_snr *= 4.0 * template.delta_f / tile_sigma
        chisq += torch.abs(tile_snr) ** 2
        dof += 2

    if dof:
        output = torch.where(active, (chisq / dof).to(output.dtype), output)
    return _torch_array(output)


class SingleDetSGChisq(SingleDetPowerChisq):
    """Class that handles precomputation and memory management for efficiently
    running the sine-Gaussian chisq
    """
    returns = {'sg_chisq': numpy.float32}

    def __init__(self, bank, num_bins=0,
                       snr_threshold=None,
                       chisq_locations=None):
        """ Create sine-Gaussian Chisq Calculator

        Parameters
        ----------
        bank: pycbc.waveform.TemplateBank
            The template bank that will be processed.
        num_bins: str
            The string determining the number of power chisq bins
        snr_threshold: float
            The threshold to calculate the sine-Gaussian chisq
        chisq_locations: list of strs
            List of strings which detail where to place a sine-Gaussian.
            The format is 'region-boolean:q1-offset1,q2-offset2'.
            The offset is relative to the end frequency of the approximant.
            The region is a boolean expression such as 'mtotal>40' indicating
            which templates to apply this set of sine-Gaussians to.
        """
        if snr_threshold is not None:
            self.do = True
            self.num_bins = num_bins
            self.snr_threshold = snr_threshold
            self.params = {}
            for descr in chisq_locations:
                region, values = descr.split(":")
                mask = bank.table.parse_boolargs([(1, region), (0, 'else')])[0]
                hashes = bank.table['template_hash'][mask.astype(bool)]
                for h in hashes:
                    self.params[h] = values
        else:
            self.do = False

    @staticmethod
    def insert_option_group(parser):
        group = parser.add_argument_group("Sine-Gaussian Chisq")
        group.add_argument("--sgchisq-snr-threshold", type=float,
            help="Minimum SNR threshold to use SG chisq")
        group.add_argument("--sgchisq-locations", type=str, nargs='+',
            help="Frequency offsets and quality factors of the sine-Gaussians"
                 " to use, format 'region-boolean:q1-offset1,q2-offset2'. "
                 "Offset is relative to the end frequency of the approximant."
                 " Region is a boolean expression selecting templates to "
                 "apply the sine-Gaussians to, ex. 'mtotal>40'")

    @classmethod
    def from_cli(cls, args, bank, chisq_bins):
        return cls(bank, chisq_bins,
                   args.sgchisq_snr_threshold,
                   args.sgchisq_locations)

    def values(self, stilde, template, psd, snrv, snr_norm,
                     bchisq, bchisq_dof, indices):
        """ Calculate sine-Gaussian chisq

        Parameters
        ----------
        stilde: pycbc.types.Frequencyseries
            The overwhitened strain
        template: pycbc.types.Frequencyseries
            The waveform template being analyzed
        psd: pycbc.types.Frequencyseries
            The power spectral density of the data
        snrv: numpy.ndarray
            The peak unnormalized complex SNR values
        snr_norm: float
            The normalization factor for the snr
        bchisq: numpy.ndarray
            The Bruce Allen power chisq values for these triggers
        bchisq_dof: numpy.ndarray
            The degrees of freedom of the Bruce chisq
        indics: numpy.ndarray
            The indices of the snr peaks.

        Returns
        -------
        chisq: Array
            Chisq values, one for each sample index
        """
        if not self.do:
            return None

        stilde_tensor = _torch_tensor(stilde)
        if template.params.template_hash not in self.params:
            if stilde_tensor is not None:
                return _torch_ones(stilde_tensor, len(snrv))
            return numpy.ones(len(snrv))
        values = self.params[template.params.template_hash].split(',')

        # Get the chisq bins to use as the frequency reference point
        bins = self.cached_chisq_bins(template, psd)

        if stilde_tensor is not None and _torch_tensor(psd) is not None:
            return _torch_sgchisq_values(
                stilde,
                template,
                psd,
                snrv,
                snr_norm,
                bchisq,
                bchisq_dof,
                indices,
                values,
                bins,
                self.snr_threshold,
            )

        # This is implemented slowly, so let's not call it often, OK?
        chisq = numpy.ones(len(snrv))
        gtem = [None for _ in values]
        for i, snrvi in enumerate(snrv):
            #Skip if newsnr too low
            snr = abs(snrvi * snr_norm)
            nsnr = ranking.newsnr(snr, bchisq[i] / bchisq_dof[i])
            if nsnr < self.snr_threshold:
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
            chisq[i] = 0

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
                    return numpy.ones(len(snrv))

                kmin = int(flow / template.delta_f)
                kmax = int(fhigh / template.delta_f)

                #Calculate sine-gaussian tile
                if gtem[idxx] is None:
                    # These are always the same values for a template, so
                    # if computing 10 sgchisq points, don't want to call
                    # this 10 times (for each SG template)
                    gtem[idxx] = sinegauss.fd_sine_gaussian(1.0, q, fcen, flow,
                                      len(template) * template.delta_f,
                                      template.delta_f).astype(numpy.complex64)
                gsigma = sigma(gtem[idxx], psd=psd,
                                     low_frequency_cutoff=flow,
                                     high_frequency_cutoff=fhigh)
                #Calculate the SNR of the tile
                gsnr = (gtem[idxx][kmin:kmax] * stilde_shift[kmin:kmax]).sum()
                gsnr *= 4.0 * gtem[idxx].delta_f / gsigma
                chisq[i] += abs(gsnr)**2.0
                dof += 2
            if dof == 0:
                chisq[i] = 1
            else:
                chisq[i] /= dof
        return chisq
