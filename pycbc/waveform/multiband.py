"""Tools for interpolating waveforms with multiple frequency bands."""

import numpy

from pycbc.types import Array, TimeSeries, zeros


def _hann_window(length, reference):
    """Return a Hann window on the same backend as ``reference``."""
    tensor = getattr(reference._data, "tensor", None)
    if tensor is None:
        return numpy.hanning(length)

    import torch
    from pycbc.types.array_torch import TorchArrayData

    if length <= 1:
        window = tensor.real.new_ones(length)
    else:
        window = torch.hann_window(
            length,
            periodic=False,
            dtype=tensor.real.dtype,
            device=tensor.device,
        )
    return Array(TorchArrayData(window), copy=False)


def multiband_fd_waveform(bands=None, lengths=None, overlap=0, **p):
    """Generate a Fourier-domain waveform using multibanding.

    Speed up generation of a Fourier-domain waveform using multibanding. This
    allows for multi-rate sampling of frequency space. Each band is
    smoothed and stitched together to produce the final waveform. The base
    approximant must support 'f_ref' and 'f_final'. The other parameters
    must be chosen carefully by the user.

    Parameters
    ----------
    bands: list or str
        The frequencies to split the waveform by. These should be chosen
        so that the corresponding length include all the waveform's frequencies
        within this band.
    lengths: list or str
        The corresponding length for each frequency band. This sets the
        resolution of the subband and should be chosen carefully so that it is
        sufficiently long to include all of the bands frequency content.
    overlap: float
        The frequency width to apply tapering between bands.
    params: dict
        The remaining keyword arguments passed to the base approximant
        waveform generation.

    Returns
    -------
    hp: pycbc.types.FrequencySeries
        Plus polarization
    hc: pycbc.types.FrequencySeries
        Cross polarization
    """
    from pycbc.waveform import get_fd_waveform

    if isinstance(bands, str):
        bands = [float(s) for s in bands.split(' ')]

    if isinstance(lengths, str):
        lengths = [float(s) for s in lengths.split(' ')]

    p['approximant'] = p['base_approximant']
    df = p['delta_f']
    fmax = p['f_final']
    flow = p['f_lower']

    bands = [flow] + bands + [fmax]
    dfs = [df] + [1.0 / duration for duration in lengths]

    dt = 1.0 / (2.0 * fmax)
    tlen = int(1.0 / dt / df)
    wf_plus = TimeSeries(zeros(tlen, dtype=numpy.float32),
                         copy=False, delta_t=dt, epoch=-1.0/df)
    wf_cross = TimeSeries(zeros(tlen, dtype=numpy.float32),
                          copy=False, delta_t=dt, epoch=-1.0/df)

    # Iterate over the sub-bands
    for i in range(len(lengths)+1):
        taper_start = taper_end = False
        if i != 0:
            taper_start = True
        if i != len(lengths):
            taper_end = True

        # Generate waveform for sub-band of full waveform
        start = bands[i]
        stop = bands[i+1]
        p2 = p.copy()
        p2['delta_f'] = dfs[i]
        p2['f_lower'] = start
        p2['f_final'] = stop

        if taper_start:
            p2['f_lower'] -= overlap / 2.0

        if taper_end:
            p2['f_final'] += overlap / 2.0

        tlen = int(1.0 / dt / dfs[i])
        flen = tlen // 2 + 1

        hp, hc = get_fd_waveform(**p2)

        # apply window function to smooth over transition regions
        kmin = int(p2['f_lower'] / dfs[i])
        kmax = int(p2['f_final'] / dfs[i])
        waves = (hp.astype(numpy.complex64), hc.astype(numpy.complex64))
        taper = _hann_window(int(overlap * 2 / dfs[i]), waves[0])

        for wf, h in zip((wf_plus, wf_cross), waves):

            if taper_start:
                h[kmin:kmin + len(taper) // 2] *= taper[:len(taper)//2]

            if taper_end:
                left = kmax - (len(taper) - len(taper) // 2)
                h[left:kmax] *= taper[len(taper)//2:]

            # add frequency band to total and use fft to interpolate
            h.resize(flen)
            h = h.to_timeseries()
            wf[len(wf)-len(h):] += h

    return (wf_plus.to_frequencyseries().astype(hp.dtype),
            wf_cross.to_frequencyseries().astype(hp.dtype))
