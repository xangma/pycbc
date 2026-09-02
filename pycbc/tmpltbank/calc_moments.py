# Copyright (C) 2013 Ian W. Harry
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import logging
import numpy

from pycbc.tmpltbank.lambda_mapping import generate_mapping

logger = logging.getLogger('pycbc.tmpltbank.calc_moments')


def _get_moments_torch(metricParams, vary_fmax=False, vary_density=None):
    """Calculate metric moments on an active CPU or CUDA Torch device.

    The metric construction that consumes these moments is NumPy based, so
    only the final scalar reductions cross back to the host.  MPS is left on
    the legacy path because these cancellation-sensitive integrals require
    float64, which MPS does not support.
    """
    from pycbc import scheme

    state = scheme.mgr.state
    if not isinstance(state, scheme.TorchScheme):
        return None

    import torch
    from pycbc.types.array_torch import (
        TorchArrayData,
        _device_matches_active,
    )

    psd_data = metricParams.psd._data
    if not isinstance(psd_data, TorchArrayData):
        return None

    psd_tensor = psd_data.tensor
    if (
        psd_tensor.device.type not in ("cpu", "cuda")
        or not _device_matches_active(psd_tensor)
        or not psd_tensor.dtype.is_floating_point
        or psd_tensor.ndim != 1
        or len(psd_tensor) < 2
    ):
        return None

    # The legacy interpolation advances its output grid by repeated scalar
    # additions.  For non-binary deltaF values, that roundoff affects which
    # source interval owns a boundary sample.  Reproduce its frequency/index
    # metadata on the host, but gather and interpolate the PSD on the device.
    source_f = (
        numpy.arange(len(psd_tensor), dtype=float) * metricParams.deltaF
    )
    new_f = []
    lower_indices = []
    frequency_differences = []
    interval_widths = []
    current_f = source_f[0]
    for i in range(len(source_f) - 1):
        f_low = source_f[i]
        f_high = source_f[i + 1]
        while current_f <= f_high:
            new_f.append(current_f)
            lower_indices.append(i)
            frequency_differences.append(current_f - f_low)
            interval_widths.append(f_high - f_low)
            current_f += metricParams.deltaF

    device = psd_tensor.device
    lower_indices = torch.as_tensor(
        lower_indices, dtype=torch.int64, device=device
    )
    frequency_differences = torch.as_tensor(
        frequency_differences, dtype=torch.float64, device=device
    )
    interval_widths = torch.as_tensor(
        interval_widths, dtype=torch.float64, device=device
    )
    psd_f = torch.as_tensor(new_f, dtype=torch.float64, device=device)

    # The legacy path promotes scalar PSD samples to Python float64 while
    # interpolating.  Preserve that precision without leaving the device.
    source_amp = psd_tensor.to(dtype=torch.float64)
    lower_amp = source_amp[lower_indices]
    gradient = (
        source_amp[lower_indices + 1] - lower_amp
    ) / interval_widths
    psd_amp = lower_amp + frequency_differences * gradient
    psd_x = psd_f / metricParams.f0
    deltax = psd_x[1] - psd_x[0]

    mask = (psd_f > metricParams.fLow) & (psd_f < metricParams.fUpper)
    psdf_red = psd_f[mask]
    psdx_red = psd_x[mask]
    base = (
        psdx_red.pow(-7.0 / 3.0)
        * deltax
        / psd_amp[mask]
    )

    cutoffs = [metricParams.fUpper]
    if vary_fmax:
        cutoffs.extend(numpy.arange(
            metricParams.fLow + vary_density,
            metricParams.fUpper,
            vary_density,
        ))

    def reduce_components(components, norm=None):
        reductions = []
        for cutoff in cutoffs:
            if cutoff == metricParams.fUpper:
                selected = components
            else:
                selected = components[psdf_red < cutoff]
            reductions.append(torch.sum(selected, dtype=torch.float64))

        # This is the intentional Torch-to-NumPy boundary: consumers build
        # small NumPy metric matrices from these scalar dictionaries.
        values = torch.stack(reductions).detach().cpu().tolist()
        moment = {}
        for cutoff, value in zip(cutoffs, values):
            if norm is not None:
                value /= norm[cutoff]
            moment[cutoff] = value
        return moment

    i7 = reduce_components(base)
    moments = {"I7": i7}

    for i in range(-7, 18):
        components = base * psdx_red.pow((-i + 7) / 3.0)
        moments[f"J{i}"] = reduce_components(components, norm=i7)

    logx = torch.log((psdx_red * metricParams.f0).pow(1.0 / 3.0))
    for power, prefix in (
        (1, "log"),
        (2, "loglog"),
        (3, "logloglog"),
        (4, "loglogloglog"),
    ):
        log_factor = logx if power == 1 else logx.pow(power)
        for i in range(-1, 18):
            components = (
                base
                * log_factor
                * psdx_red.pow((-i + 7) / 3.0)
            )
            moments[f"{prefix}{i}"] = reduce_components(
                components, norm=i7
            )

    return moments


def determine_eigen_directions(metricParams, preserveMoments=False,
                               vary_fmax=False, vary_density=None):
    """
    This function will calculate the coordinate transfomations that are needed
    to rotate from a coordinate system described by the various Lambda
    components in the frequency expansion, to a coordinate system where the
    metric is Cartesian.

    Parameters
    -----------
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric.
    preserveMoments : boolean, optional (default False)
        Currently only used for debugging.
        If this is given then if the moments structure is already set
        within metricParams then they will not be recalculated.
    vary_fmax : boolean, optional (default False)
        If set to False the metric and rotations are calculated once, for the
        full range of frequency [f_low,f_upper).
        If set to True the metric and rotations are calculated multiple times,
        for frequency ranges [f_low,f_low + i*vary_density), where i starts at
        1 and runs up until f_low + (i+1)*vary_density > f_upper.
        Thus values greater than f_upper are *not* computed.
        The calculation for the full range [f_low,f_upper) is also done.
    vary_density : float, optional
        If vary_fmax is True, this will be used in computing the frequency
        ranges as described for vary_fmax.

    Returns
    --------
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric.
        **THIS FUNCTION ONLY RETURNS THE CLASS**
        The following will be **added** to this structure
    metricParams.evals : Dictionary of numpy.array
        Each entry in the dictionary corresponds to the different frequency
        ranges described in vary_fmax. If vary_fmax = False, the only entry
        will be f_upper, this corresponds to integrals in [f_low,f_upper). This
        entry is always present. Each other entry will use floats as keys to
        the dictionary. These floats give the upper frequency cutoff when it is
        varying.
        Each numpy.array contains the eigenvalues which, with the eigenvectors
        in evecs, are needed to rotate the
        coordinate system to one in which the metric is the identity matrix.
    metricParams.evecs : Dictionary of numpy.matrix
        Each entry in the dictionary is as described under evals.
        Each numpy.matrix contains the eigenvectors which, with the eigenvalues
        in evals, are needed to rotate the
        coordinate system to one in which the metric is the identity matrix.
    metricParams.metric : Dictionary of numpy.matrix
        Each entry in the dictionary is as described under evals.
        Each numpy.matrix contains the metric of the parameter space in the
        Lambda_i coordinate system.
    metricParams.moments : Moments structure
        See the structure documentation for a description of this. This
        contains the result of all the integrals used in computing the metrics
        above. It can be used for the ethinca components calculation, or other
        similar calculations.
    """

    evals = {}
    evecs = {}
    metric = {}
    unmax_metric = {}

    # First step is to get the moments needed to calculate the metric
    if not (metricParams.moments and preserveMoments):
        get_moments(metricParams, vary_fmax=vary_fmax,
                    vary_density=vary_density)

    # What values are going to be in the moments
    # J7 is the normalization factor so it *MUST* be present
    list = metricParams.moments['J7'].keys()

    # We start looping over every item in the list of metrics
    for item in list:
        # Here we convert the moments into a form easier to use here
        Js = {}
        for i in range(-7,18):
            Js[i] = metricParams.moments['J%d'%(i)][item]

        logJs = {}
        for i in range(-1,18):
            logJs[i] = metricParams.moments['log%d'%(i)][item]

        loglogJs = {}
        for i in range(-1,18):
            loglogJs[i] = metricParams.moments['loglog%d'%(i)][item]

        logloglogJs = {}
        for i in range(-1,18):
            logloglogJs[i] = metricParams.moments['logloglog%d'%(i)][item]

        loglogloglogJs = {}
        for i in range(-1,18):
            loglogloglogJs[i] = metricParams.moments['loglogloglog%d'%(i)][item]

        mapping = generate_mapping(metricParams.pnOrder)

        # Calculate the metric
        gs, unmax_metric_curr = calculate_metric(Js, logJs, loglogJs,
                                          logloglogJs, loglogloglogJs, mapping)
        metric[item] = gs
        unmax_metric[item] = unmax_metric_curr

        # And the eigenvalues
        evals[item], evecs[item] = numpy.linalg.eig(gs)

        # Numerical error can lead to small negative eigenvalues.
        for i in range(len(evals[item])):
            if evals[item][i] < 0:
                # Due to numerical imprecision the very small eigenvalues can
                # be negative. Make these positive.
                evals[item][i] = -evals[item][i]
            if evecs[item][i,i] < 0:
                # We demand a convention that all diagonal terms in the matrix
                # of eigenvalues are positive.
                # This is done to help visualization of the spaces (increasing
                # mchirp always goes the same way)
                evecs[item][:,i] = - evecs[item][:,i]

    metricParams.evals = evals
    metricParams.evecs = evecs
    metricParams.metric = metric
    metricParams.time_unprojected_metric = unmax_metric

    return metricParams

def get_moments(metricParams, vary_fmax=False, vary_density=None):
    """
    This function will calculate the various integrals (moments) that are
    needed to compute the metric used in template bank placement and
    coincidence.

    Parameters
    -----------
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric.
    vary_fmax : boolean, optional (default False)
        If set to False the metric and rotations are calculated once, for the
        full range of frequency [f_low,f_upper).
        If set to True the metric and rotations are calculated multiple times,
        for frequency ranges [f_low,f_low + i*vary_density), where i starts at
        1 and runs up until f_low + (i+1)*vary_density > f_upper.
        Thus values greater than f_upper are *not* computed.
        The calculation for the full range [f_low,f_upper) is also done.
    vary_density : float, optional
        If vary_fmax is True, this will be used in computing the frequency
        ranges as described for vary_fmax.

    Returns
    --------
    None : None
        **THIS FUNCTION RETURNS NOTHING**
        The following will be **added** to the metricParams structure
    metricParams.moments : Moments structure
        This contains the result of all the integrals used in computing the
        metrics above. It can be used for the ethinca components calculation,
        or other similar calculations. This is composed of two compound
        dictionaries. The first entry indicates which moment is being
        calculated and the second entry indicates the upper frequency cutoff
        that was used.

        In all cases x = f/f0.

        For the first entries the options are:

        moments['J%d' %(i)][f_cutoff]
        This stores the integral of
        x**((-i)/3.) * delta X / PSD(x)

        moments['log%d' %(i)][f_cutoff]
        This stores the integral of
        (numpy.log(x**(1./3.))) x**((-i)/3.) * delta X / PSD(x)

        moments['loglog%d' %(i)][f_cutoff]
        This stores the integral of
        (numpy.log(x**(1./3.)))**2 x**((-i)/3.) * delta X / PSD(x)

        moments['loglog%d' %(i)][f_cutoff]
        This stores the integral of
        (numpy.log(x**(1./3.)))**3 x**((-i)/3.) * delta X / PSD(x)

        moments['loglog%d' %(i)][f_cutoff]
        This stores the integral of
        (numpy.log(x**(1./3.)))**4 x**((-i)/3.) * delta X / PSD(x)

        The second entry stores the frequency cutoff used when computing
        the integral. See description of the vary_fmax option above.

        All of these values are nomralized by a factor of

        x**((-7)/3.) * delta X / PSD(x)

        The normalization factor can be obtained in

        moments['I7'][f_cutoff]
    """
    # NOTE: Unless the TaylorR2F4 metric is used the log^3 and log^4 terms are
    # not needed. As this calculation is not too slow compared to bank
    # placement we just do this anyway.

    moments = _get_moments_torch(
        metricParams,
        vary_fmax=vary_fmax,
        vary_density=vary_density,
    )
    if moments is not None:
        metricParams.moments = moments
        return

    psd_amp = metricParams.psd.data
    psd_f = numpy.arange(len(psd_amp), dtype=float) * metricParams.deltaF
    new_f, new_amp = interpolate_psd(psd_f, psd_amp, metricParams.deltaF)

    # Need I7 first as this is the normalization factor
    funct = lambda x,f0: 1
    I7 = calculate_moment(new_f, new_amp, metricParams.fLow, \
                          metricParams.fUpper, metricParams.f0, funct,\
                          vary_fmax=vary_fmax, vary_density=vary_density)

    # Do all the J moments
    moments = {}
    moments['I7'] = I7
    for i in range(-7,18):
        funct = lambda x,f0: x**((-i+7)/3.)
        moments['J%d' %(i)] = calculate_moment(new_f, new_amp, \
                                metricParams.fLow, metricParams.fUpper, \
                                metricParams.f0, funct, norm=I7, \
                                vary_fmax=vary_fmax, vary_density=vary_density)

    # Do the logx multiplied by some power terms
    for i in range(-1,18):
        funct = lambda x,f0: (numpy.log((x*f0)**(1./3.))) * x**((-i+7)/3.)
        moments['log%d' %(i)] = calculate_moment(new_f, new_amp, \
                                metricParams.fLow, metricParams.fUpper, \
                                metricParams.f0, funct, norm=I7, \
                                vary_fmax=vary_fmax, vary_density=vary_density)

    # Do the loglog term
    for i in range(-1,18):
        funct = lambda x,f0: (numpy.log((x*f0)**(1./3.)))**2 * x**((-i+7)/3.)
        moments['loglog%d' %(i)] = calculate_moment(new_f, new_amp, \
                                metricParams.fLow, metricParams.fUpper, \
                                metricParams.f0, funct, norm=I7, \
                                vary_fmax=vary_fmax, vary_density=vary_density)

    # Do the logloglog term
    for i in range(-1,18):
        funct = lambda x,f0: (numpy.log((x*f0)**(1./3.)))**3 * x**((-i+7)/3.)
        moments['logloglog%d' %(i)] = calculate_moment(new_f, new_amp, \
                                metricParams.fLow, metricParams.fUpper, \
                                metricParams.f0, funct, norm=I7, \
                                vary_fmax=vary_fmax, vary_density=vary_density)

    # Do the logloglog term
    for i in range(-1,18):
        funct = lambda x,f0: (numpy.log((x*f0)**(1./3.)))**4 * x**((-i+7)/3.)
        moments['loglogloglog%d' %(i)] = calculate_moment(new_f, new_amp, \
                                metricParams.fLow, metricParams.fUpper, \
                                metricParams.f0, funct, norm=I7, \
                                vary_fmax=vary_fmax, vary_density=vary_density)

    metricParams.moments = moments

def interpolate_psd(psd_f, psd_amp, deltaF):
    """
    Function to interpolate a PSD to a different value of deltaF. Uses linear
    interpolation.

    Parameters
    ----------
    psd_f : numpy.array or list or similar
        List of the frequencies contained within the PSD.
    psd_amp : numpy.array or list or similar
        List of the PSD values at the frequencies in psd_f.
    deltaF : float
        Value of deltaF to interpolate the PSD to.

    Returns
    --------
    new_psd_f : numpy.array
       Array of the frequencies contained within the interpolated PSD
    new_psd_amp : numpy.array
       Array of the interpolated PSD values at the frequencies in new_psd_f.
    """
    # In some cases this will be a no-op. I thought about removing this, but
    # this function can take unequally sampled PSDs and it is difficult to
    # check for this. As this function runs quickly anyway (compared to the
    # moment calculation) I decided to always interpolate.

    new_psd_f = []
    new_psd_amp = []
    fcurr = psd_f[0]

    for i in range(len(psd_f) - 1):
        f_low = psd_f[i]
        f_high = psd_f[i+1]
        amp_low = psd_amp[i]
        amp_high = psd_amp[i+1]
        while(1):
            if fcurr > f_high:
                break
            new_psd_f.append(fcurr)
            gradient = (amp_high - amp_low) / (f_high - f_low)
            fDiff = fcurr - f_low
            new_psd_amp.append(amp_low + fDiff * gradient)
            fcurr = fcurr + deltaF
    return numpy.asarray(new_psd_f), numpy.asarray(new_psd_amp)


def calculate_moment(psd_f, psd_amp, fmin, fmax, f0, funct,
                     norm=None, vary_fmax=False, vary_density=None):
    r"""
    Function for calculating one of the integrals used to construct a template
    bank placement metric. The integral calculated will be

    \int funct(x) * (psd_x)**(-7./3.) * delta_x / PSD(x)

    where x = f / f0. The lower frequency cutoff is given by fmin, see
    the parameters below for details on how the upper frequency cutoff is
    chosen

    Parameters
    -----------
    psd_f : numpy.array
       numpy array holding the set of evenly spaced frequencies used in the PSD
    psd_amp : numpy.array
       numpy array holding the PSD values corresponding to the psd_f
       frequencies
    fmin : float
        The lower frequency cutoff used in the calculation of the integrals
        used to obtain the metric.
    fmax : float
        The upper frequency cutoff used in the calculation of the integrals
        used to obtain the metric. This can be varied (see the vary_fmax
        option below).
    f0 : float
        This is an arbitrary scaling factor introduced to avoid the potential
        for numerical overflow when calculating this. Generally the default
        value (70) is safe here. **IMPORTANT, if you want to calculate the
        ethinca metric components later this MUST be set equal to f_low.**
    funct : Lambda function
        The function to use when computing the integral as described above.
    norm : Dictionary of floats
        If given then moment[f_cutoff] will be divided by norm[f_cutoff]
    vary_fmax : boolean, optional (default False)
        If set to False the metric and rotations are calculated once, for the
        full range of frequency [f_low,f_upper).
        If set to True the metric and rotations are calculated multiple times,
        for frequency ranges [f_low,f_low + i*vary_density), where i starts at
        1 and runs up until f_low + (i+1)*vary_density > f_upper.
        Thus values greater than f_upper are *not* computed.
        The calculation for the full range [f_low,f_upper) is also done.
    vary_density : float, optional
        If vary_fmax is True, this will be used in computing the frequency
        ranges as described for vary_fmax.

    Returns
    --------
    moment : Dictionary of floats
        moment[f_cutoff] will store the value of the moment at the frequency
        cutoff given by f_cutoff.
    """

    # Must ensure deltaF in psd_f is constant
    psd_x = psd_f / f0
    deltax = psd_x[1] - psd_x[0]

    mask = numpy.logical_and(psd_f > fmin, psd_f < fmax)
    psdf_red = psd_f[mask]
    comps_red = psd_x[mask] ** (-7./3.) * funct(psd_x[mask], f0) * deltax / \
                psd_amp[mask]
    moment = {}
    moment[fmax] = comps_red.sum()
    if norm:
        moment[fmax] = moment[fmax] / norm[fmax]
    if vary_fmax:
        for t_fmax in numpy.arange(fmin + vary_density, fmax, vary_density):
            moment[t_fmax] = comps_red[psdf_red < t_fmax].sum()
            if norm:
                moment[t_fmax] = moment[t_fmax] / norm[t_fmax]
    return moment

def calculate_metric(Js, logJs, loglogJs, logloglogJs, loglogloglogJs, \
                     mapping):
    """
    This function will take the various integrals calculated by get_moments and
    convert this into a metric for the appropriate parameter space.

    Parameters
    -----------
    Js : Dictionary
        The list of (log^0 x) * x**(-i/3) integrals computed by get_moments()
        The index is Js[i]
    logJs : Dictionary
        The list of (log^1 x) * x**(-i/3) integrals computed by get_moments()
        The index is logJs[i]
    loglogJs : Dictionary
        The list of (log^2 x) * x**(-i/3) integrals computed by get_moments()
        The index is loglogJs[i]
    logloglogJs : Dictionary
        The list of (log^3 x) * x**(-i/3) integrals computed by get_moments()
        The index is logloglogJs[i]
    loglogloglogJs : Dictionary
        The list of (log^4 x) * x**(-i/3) integrals computed by get_moments()
        The index is loglogloglogJs[i]
    mapping : dictionary
        Used to identify which Lambda components are active in this parameter
        space and map these to entries in the metric matrix.

    Returns
    --------
    metric : numpy.matrix
        The resulting metric.
    """

    # How many dimensions in the parameter space?
    maxLen = len(mapping.keys())

    metric = numpy.zeros(shape=(maxLen,maxLen), dtype=float)
    unmax_metric = numpy.zeros(shape=(maxLen+1,maxLen+1), dtype=float)

    for i in range(16):
        for j in range(16):
            calculate_metric_comp(metric, unmax_metric, i, j, Js,
                                           logJs, loglogJs, logloglogJs,
                                           loglogloglogJs, mapping)
    return metric, unmax_metric


def calculate_metric_comp(gs, unmax_metric, i, j, Js, logJs, loglogJs,
                          logloglogJs, loglogloglogJs, mapping):
    """
    Used to compute part of the metric. Only call this from within
    calculate_metric(). Please see the documentation for that function.
    """
    # Time term in unmax_metric. Note that these terms are recomputed a bunch
    # of time, but this cost is insignificant compared to computing the moments
    unmax_metric[-1,-1] = (Js[1] - Js[4]*Js[4])

    # Normal terms
    if 'Lambda%d'%i in mapping and 'Lambda%d'%j in mapping:
        gammaij = Js[17-i-j] - Js[12-i]*Js[12-j]
        gamma0i = (Js[9-i] - Js[4]*Js[12-i])
        gamma0j = (Js[9-j] - Js[4] * Js[12-j])
        gs[mapping['Lambda%d'%i],mapping['Lambda%d'%j]] = \
            0.5 * (gammaij - gamma0i*gamma0j/(Js[1] - Js[4]*Js[4]))
        unmax_metric[mapping['Lambda%d'%i], -1] = gamma0i
        unmax_metric[-1, mapping['Lambda%d'%j]] = gamma0j
        unmax_metric[mapping['Lambda%d'%i],mapping['Lambda%d'%j]] = gammaij
    # Normal,log cross terms
    if 'Lambda%d'%i in mapping and 'LogLambda%d'%j in mapping:
        gammaij = logJs[17-i-j] - logJs[12-j] * Js[12-i]
        gamma0i = (Js[9-i] - Js[4] * Js[12-i])
        gamma0j = logJs[9-j] - logJs[12-j] * Js[4]
        gs[mapping['Lambda%d'%i],mapping['LogLambda%d'%j]] = \
            gs[mapping['LogLambda%d'%j],mapping['Lambda%d'%i]] = \
            0.5 * (gammaij - gamma0i*gamma0j/(Js[1] - Js[4]*Js[4]))
        unmax_metric[mapping['Lambda%d'%i], -1] = gamma0i
        unmax_metric[-1, mapping['Lambda%d'%i]] = gamma0i
        unmax_metric[-1, mapping['LogLambda%d'%j]] = gamma0j
        unmax_metric[mapping['LogLambda%d'%j], -1] = gamma0j
        unmax_metric[mapping['Lambda%d'%i],mapping['LogLambda%d'%j]] = gammaij
        unmax_metric[mapping['LogLambda%d'%j],mapping['Lambda%d'%i]] = gammaij
    # Log,log terms
    if 'LogLambda%d'%i in mapping and 'LogLambda%d'%j in mapping:
        gammaij = loglogJs[17-i-j] - logJs[12-j] * logJs[12-i]
        gamma0i = (logJs[9-i] - Js[4] * logJs[12-i])
        gamma0j = logJs[9-j] - logJs[12-j] * Js[4]
        gs[mapping['LogLambda%d'%i],mapping['LogLambda%d'%j]] = \
            0.5 * (gammaij - gamma0i*gamma0j/(Js[1] - Js[4]*Js[4]))
        unmax_metric[mapping['LogLambda%d'%i], -1] = gamma0i
        unmax_metric[-1, mapping['LogLambda%d'%j]] = gamma0j
        unmax_metric[mapping['LogLambda%d'%i],mapping['LogLambda%d'%j]] =\
            gammaij

    # Normal,loglog cross terms
    if 'Lambda%d'%i in mapping and 'LogLogLambda%d'%j in mapping:
        gammaij = loglogJs[17-i-j] - loglogJs[12-j] * Js[12-i]
        gamma0i = (Js[9-i] - Js[4] * Js[12-i])
        gamma0j = loglogJs[9-j] - loglogJs[12-j] * Js[4]
        gs[mapping['Lambda%d'%i],mapping['LogLogLambda%d'%j]] = \
            gs[mapping['LogLogLambda%d'%j],mapping['Lambda%d'%i]] = \
            0.5 * (gammaij - gamma0i*gamma0j/(Js[1] - Js[4]*Js[4]))
        unmax_metric[mapping['Lambda%d'%i], -1] = gamma0i
        unmax_metric[-1, mapping['Lambda%d'%i]] = gamma0i
        unmax_metric[-1, mapping['LogLogLambda%d'%j]] = gamma0j
        unmax_metric[mapping['LogLogLambda%d'%j], -1] = gamma0j
        unmax_metric[mapping['Lambda%d'%i],mapping['LogLogLambda%d'%j]] = \
            gammaij
        unmax_metric[mapping['LogLogLambda%d'%j],mapping['Lambda%d'%i]] = \
            gammaij

    # log,loglog cross terms
    if 'LogLambda%d'%i in mapping and 'LogLogLambda%d'%j in mapping:
        gammaij = logloglogJs[17-i-j] - loglogJs[12-j] * logJs[12-i]
        gamma0i = (logJs[9-i] - Js[4] * logJs[12-i])
        gamma0j = loglogJs[9-j] - loglogJs[12-j] * Js[4]
        gs[mapping['LogLambda%d'%i],mapping['LogLogLambda%d'%j]] = \
            gs[mapping['LogLogLambda%d'%j],mapping['LogLambda%d'%i]] = \
            0.5 * (gammaij - gamma0i*gamma0j/(Js[1] - Js[4]*Js[4]))
        unmax_metric[mapping['LogLambda%d'%i], -1] = gamma0i
        unmax_metric[-1, mapping['LogLambda%d'%i]] = gamma0i
        unmax_metric[-1, mapping['LogLogLambda%d'%j]] = gamma0j
        unmax_metric[mapping['LogLogLambda%d'%j], -1] = gamma0j
        unmax_metric[mapping['LogLambda%d'%i],mapping['LogLogLambda%d'%j]] = \
            gammaij
        unmax_metric[mapping['LogLogLambda%d'%j],mapping['LogLambda%d'%i]] = \
            gammaij

    # Loglog,loglog terms
    if 'LogLogLambda%d'%i in mapping and 'LogLogLambda%d'%j in mapping:
        gammaij = loglogloglogJs[17-i-j] - loglogJs[12-j] * loglogJs[12-i]
        gamma0i = (loglogJs[9-i] - Js[4] * loglogJs[12-i])
        gamma0j = loglogJs[9-j] - loglogJs[12-j] * Js[4]
        gs[mapping['LogLogLambda%d'%i],mapping['LogLogLambda%d'%j]] = \
            0.5 * (gammaij - gamma0i*gamma0j/(Js[1] - Js[4]*Js[4]))
        unmax_metric[mapping['LogLogLambda%d'%i], -1] = gamma0i
        unmax_metric[-1, mapping['LogLogLambda%d'%j]] = gamma0j
        unmax_metric[mapping['LogLogLambda%d'%i],mapping['LogLogLambda%d'%j]] =\
            gammaij
