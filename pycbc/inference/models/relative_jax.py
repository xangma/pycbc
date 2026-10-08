"""JAX relative-binning model orchestration."""


import itertools


import logging


import numpy


from pycbc.detector import Detector


from pycbc.types import Array, TimeSeries


from pycbc.types.backend import wrap_backend_array


from pycbc.waveform import fd_det_sequence, get_fd_det_waveform_sequence, get_fd_waveform_sequence


from .gaussian_noise import catch_waveform_error


from .relbin_cpu import likelihood_parts_det, likelihood_parts_multi_v, likelihood_parts_v, likelihood_parts_vector, likelihood_parts_vectorp, likelihood_parts_vectort


from .proposals_jax import _jax_array, _threshold_extent


from .relbin import Relative, setup_bins


from .gaussian_noise_jax import JAXBaseGaussianNoise


from .proposals_jax import JAXDistMarg


from pycbc.waveform.generator_jax import _radiation_parameters


def _numpy_value(value):
    """Return a CPU representation for the legacy relative-bin kernels."""
    arr = _jax_array(value)
    if arr is not None:
        return numpy.asarray(arr)
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _time_series_from_values(values, delta_t, epoch):
    """Wrap values as a TimeSeries without copying backend storage."""
    arr = _jax_array(values)
    if arr is not None:
        return TimeSeries(
            wrap_backend_array(arr), delta_t=delta_t, epoch=epoch, copy=False
        )
    return TimeSeries(values, delta_t=delta_t, epoch=epoch)


def _prepare_reference_data(waveform, data, size, offset, delta_f, time_shift):
    """Place a fiducial waveform and shifted data on their active backend."""
    if any(_jax_array(value) is not None for value in (waveform, data)):
        from .relbin_jax import prepare_reference_data

        return prepare_reference_data(waveform, data, size, offset, delta_f, time_shift)

    waveform.resize(size)
    waveform = numpy.roll(waveform, offset)
    frequencies = numpy.arange(size, dtype=numpy.float64) * delta_f
    shift = numpy.exp(-2.0j * numpy.pi * frequencies * time_shift)
    return numpy.array(waveform), data * numpy.conjugate(shift)


def _uniform_frequency_grid(series):
    """Rebuild a frequency grid without copying its device representation."""
    return numpy.arange(len(series), dtype=numpy.float64) * series.delta_f


class JAXRelative(Relative, JAXDistMarg, JAXBaseGaussianNoise):
    """JAX implementation of :class:`Relative`."""

    def __init__(
        self,
        variable_params,
        data,
        low_frequency_cutoff,
        fiducial_params=None,
        gammas=None,
        epsilon=0.5,
        earth_rotation=False,
        earth_rotation_mode=2,
        marginalize_phase=True,
        **kwargs,
    ):
        variable_params, kwargs = self.setup_marginalization(
            variable_params, marginalize_phase=marginalize_phase, **kwargs
        )

        super(Relative, self).__init__(
            variable_params, data, low_frequency_cutoff, **kwargs
        )

        # If the waveform needs us to apply the detector response,
        # set flag to true (most cases for ground-based observatories).
        self.still_needs_det_response = False
        if self.static_params["approximant"] in fd_det_sequence:
            self.still_needs_det_response = True

        # reference waveform and bin edges
        self.f, self.df, self.end_time, self.det = {}, {}, {}, {}
        self.h00, self.h00_sparse = {}, {}
        self.fedges, self.edges = {}, {}
        self.ta, self.antenna_time = {}, {}

        # filtered summary data for linear approximation
        self.sdat = {}
        self._jax_likelihood_cache = {}
        self._jax_multi_likelihood_cache = {}

        # store fiducial waveform params
        self.fid_params = self.static_params.copy()
        self.fid_params.update(fiducial_params)

        # the flag used in `_loglr`
        self.return_sh_hh = False

        for k in self.static_params:
            if self.fid_params[k] == "REPLACE":
                self.fid_params.pop(k)

        for ifo in data:
            # store data and frequencies
            d0 = self.data[ifo]
            self.df[ifo] = d0.delta_f
            self.f[ifo] = _uniform_frequency_grid(d0)
            self.end_time[ifo] = float(d0.end_time)

            # generate fiducial waveform
            f_lo = self.kmin[ifo] * self.df[ifo]
            f_hi = self.kmax[ifo] * self.df[ifo]
            logging.info(
                "%s: Generating fiducial waveform from %s to %s Hz",
                ifo,
                f_lo,
                f_hi,
            )

            # prune low frequency samples to avoid waveform errors
            fpoints = Array(self.f[ifo].astype(numpy.float64))
            fpoints = fpoints[self.kmin[ifo] : self.kmax[ifo] + 1]

            if self.still_needs_det_response:
                wave = get_fd_det_waveform_sequence(
                    ifos=ifo, sample_points=fpoints, **self.fid_params
                )
                curr_wav = wave[ifo]
                self.ta[ifo] = 0.0
            else:
                fid_hp, fid_hc = get_fd_waveform_sequence(
                    sample_points=fpoints, **_radiation_parameters(self.fid_params)
                )
                # Apply detector response if not handled by
                # the waveform generator
                self.det[ifo] = Detector(ifo)
                dt = self.det[ifo].time_delay_from_earth_center(
                    self.fid_params["ra"],
                    self.fid_params["dec"],
                    self.fid_params["tc"],
                )
                self.ta[ifo] = self.fid_params["tc"] + dt
                fp, fc = self.det[ifo].antenna_pattern(
                    self.fid_params["ra"],
                    self.fid_params["dec"],
                    self.fid_params["polarization"],
                    self.fid_params["tc"],
                )
                curr_wav = fid_hp * fp + fid_hc * fc

            # check for zeros at low and high frequencies
            # make sure only nonzero samples are included in bins
            try:
                first_nonzero, last_nonzero = _threshold_extent(curr_wav, 0.0)
            except IndexError as exc:
                # Preserve the legacy ``list.index(True)`` failure for an
                # entirely zero fiducial waveform.
                raise ValueError("True is not in list") from exc
            numzeros_lo = first_nonzero
            if numzeros_lo > 0:
                new_kmin = self.kmin[ifo] + numzeros_lo
                f_lo = new_kmin * self.df[ifo]
                logging.info(
                    "WARNING! Fiducial waveform starts above "
                    "low-frequency-cutoff, initial bin frequency "
                    "will be %s Hz",
                    f_lo,
                )
            numzeros_hi = len(curr_wav) - last_nonzero - 1
            if numzeros_hi > 0:
                new_kmax = self.kmax[ifo] - numzeros_hi
                f_hi = new_kmax * self.df[ifo]
                logging.info(
                    "WARNING! Fiducial waveform terminates below "
                    "high-frequency-cutoff, final bin frequency "
                    "will be %s Hz",
                    f_hi,
                )

            self.ta[ifo] -= self.end_time[ifo]
            # Apply the time shift to the data in lieu of the reference
            # waveform. This makes target/reference comparisons simpler.
            self.h00[ifo], data_shifted = _prepare_reference_data(
                curr_wav,
                self.data[ifo],
                len(self.f[ifo]),
                self.kmin[ifo],
                self.df[ifo],
                self.ta[ifo],
            )

            logging.info("Computing frequency bins")
            fbin_ind = setup_bins(
                f_full=self.f[ifo],
                f_lo=f_lo,
                f_hi=f_hi,
                gammas=gammas,
                eps=float(epsilon),
            )
            logging.info("Using %s bins for this model", len(fbin_ind))

            self.fedges[ifo] = self.f[ifo][fbin_ind]
            self.edges[ifo] = fbin_ind
            self.init_from_frequencies(data_shifted, self.h00, fbin_ind, ifo)
            self.antenna_time[ifo] = self.setup_antenna(
                earth_rotation, int(earth_rotation_mode), self.fedges[ifo]
            )
        self.combine_layout()

    def combine_layout(self):
        # determine the unique ifo layouts
        self.edge_unique = []
        self.ifo_map = {}
        unique_layouts = []
        for ifo in self.fedges:
            for i, layout in enumerate(unique_layouts):
                if numpy.array_equal(layout, self.fedges[ifo]):
                    self.ifo_map[ifo] = i
                    break
            else:
                self.ifo_map[ifo] = len(self.edge_unique)
                unique_layouts.append(self.fedges[ifo])
                self.edge_unique.append(Array(self.fedges[ifo]))
        logging.info("%s unique ifo layouts", len(self.edge_unique))

    def summary_product(self, h1, h2, bins, ifo):
        """Calculate the summary values for the inner product <h1|h2>"""
        psd = self.psds[ifo]
        if any(_jax_array(value) is not None for value in (h1, h2, psd)):
            from .relbin_jax import summary_product

            return summary_product(h1, h2, psd, self.f[ifo], bins, self.df[ifo])

        # calculate coefficients
        h12 = numpy.conjugate(h1) * h2 / psd

        # constant terms
        a0 = numpy.array(
            [4.0 * self.df[ifo] * h12[low:high].sum() for low, high in bins]
        )

        # linear terms
        a1 = numpy.array(
            [
                4.0
                / (high - low)
                * (h12[low:high] * (self.f[ifo][low:high] - self.f[ifo][low])).sum()
                for low, high in bins
            ]
        )

        return a0, a1

    def _get_jax_likelihood_data(self, ifo, waveform):
        """Cache static likelihood inputs beside a JAX waveform."""
        arr = _jax_array(waveform)
        if arr is None:
            return None

        cache_key = (arr.dtype, arr.device)
        cache = self._jax_likelihood_cache.setdefault(ifo, {})
        if cache_key not in cache:
            from .relbin_jax import prepare_likelihood_data

            sdat = self.sdat[ifo]
            cache[cache_key] = prepare_likelihood_data(
                arr,
                self.fedges[ifo],
                self.h00_sparse[ifo],
                sdat["a0"],
                sdat["a1"],
                sdat["b0"],
                sdat["b1"],
            )
        return cache[cache_key]

    def _get_jax_multi_likelihood_data(
        self, m1, m2, ifo, waveform, waveform2, freqs, h00, h002, a0, a1
    ):
        """Cache static multi-signal inputs beside JAX waveforms."""
        arr = _jax_array(waveform)
        if arr is None or _jax_array(waveform2) is None:
            return None

        pair_cache = self._jax_multi_likelihood_cache.setdefault((m1, m2, ifo), {})
        cache_key = (arr.dtype, arr.device)
        if cache_key not in pair_cache:
            from .relbin_jax import prepare_multi_likelihood_data

            pair_cache[cache_key] = prepare_multi_likelihood_data(
                arr, freqs, h00, h002, a0, a1
            )
        return pair_cache[cache_key]

    def get_waveforms(self, params, keep_backend=False):
        """Get the waveform polarizations for each ifo"""
        keep = keep_backend
        if self.still_needs_det_response:
            wfs = {}
            for ifo in self.data:
                wfs.update(
                    get_fd_det_waveform_sequence(
                        ifos=ifo, sample_points=self.fedges[ifo], **params
                    )
                )
            return wfs

        wfs = []
        for edge in self.edge_unique:
            hp, hc = get_fd_waveform_sequence(
                sample_points=edge, **_radiation_parameters(params)
            )
            if not keep or _jax_array(hp) is None:
                hp = hp.numpy()
                hc = hc.numpy()
            wfs.append((hp, hc))
        wf_ret = {ifo: wfs[self.ifo_map[ifo]] for ifo in self.data}

        self.wf_ret = wf_ret
        return wf_ret

    def _polarization_likelihood_parts(self, ifo, params, waveform, likelihood):
        """Evaluate one detector's polarization likelihood products."""
        hp, hc = waveform
        freqs = self.fedges[ifo]
        sdat = self.sdat[ifo]
        h00 = self.h00_sparse[ifo]
        times = self.antenna_time[ifo]
        detector = self.det[ifo]
        jax_data = self._get_jax_likelihood_data(ifo, hp)
        if jax_data is not None:
            from . import relbin_jax

            fp, fc, delay = relbin_jax.detector_response(
                detector, params["ra"], params["dec"], times, hp
            )
        else:
            fp, fc = detector.antenna_pattern(params["ra"], params["dec"], 0.0, times)
            delay = detector.time_delay_from_earth_center(
                params["ra"], params["dec"], times
            )
        earth_time = self.lformat in ("earth_time", "earth_time_pol")
        if earth_time:
            dtc = params["tc"] - self.end_time[ifo] - self.ta[ifo]
        else:
            dtc = params["tc"] + delay - self.end_time[ifo] - self.ta[ifo]
        if jax_data is not None:
            freqs_j, h00_j, a0_j, a1_j, b0_j, b1_j = jax_data
            pol_phase = relbin_jax.polarization_phase(params["polarization"], hp)
            if self.lformat == "earth_time_pol":
                filt, norm = relbin_jax.likelihood_parts_v_pol_time(
                    freqs_j,
                    fp,
                    fc,
                    delay,
                    dtc,
                    pol_phase,
                    hp,
                    hc,
                    h00_j,
                    a0_j,
                    a1_j,
                    b0_j,
                    b1_j,
                )
            elif self.lformat == "earth_pol":
                filt, norm = relbin_jax.likelihood_parts_v_pol(
                    freqs_j,
                    fp,
                    fc,
                    dtc,
                    pol_phase,
                    hp,
                    hc,
                    h00_j,
                    a0_j,
                    a1_j,
                    b0_j,
                    b1_j,
                )
            else:
                fp, fc = relbin_jax.polarized_antenna_response(fp, fc, pol_phase, hp)
                if self.lformat == "earth_time":
                    filt, norm = relbin_jax.likelihood_parts_v_time(
                        freqs_j,
                        fp,
                        fc,
                        delay,
                        dtc,
                        hp,
                        hc,
                        h00_j,
                        a0_j,
                        a1_j,
                        b0_j,
                        b1_j,
                    )
                elif likelihood in (
                    likelihood_parts_vector,
                    likelihood_parts_vectort,
                    likelihood_parts_vectorp,
                ):
                    filt, norm = relbin_jax.likelihood_parts_vector(
                        freqs_j, fp, fc, dtc, hp, hc, h00_j, a0_j, a1_j, b0_j, b1_j
                    )
                elif likelihood is likelihood_parts_v:
                    filt, norm = relbin_jax.likelihood_parts_v(
                        freqs_j, fp, fc, dtc, hp, hc, h00_j, a0_j, a1_j, b0_j, b1_j
                    )
                else:
                    filt, norm = relbin_jax.likelihood_parts(
                        freqs_j, fp, fc, dtc, hp, hc, h00_j, a0_j, a1_j, b0_j, b1_j
                    )
        else:
            hp, hc = _numpy_value(hp), _numpy_value(hc)
            pol_phase = numpy.exp(-2.0j * params["polarization"])
            if self.lformat == "earth_time_pol":
                filt, norm = likelihood(
                    freqs,
                    fp,
                    fc,
                    delay,
                    dtc,
                    pol_phase,
                    hp,
                    hc,
                    h00,
                    sdat["a0"],
                    sdat["a1"],
                    sdat["b0"],
                    sdat["b1"],
                )
            elif self.lformat == "earth_pol":
                filt, norm = likelihood(
                    freqs,
                    fp,
                    fc,
                    dtc,
                    pol_phase,
                    hp,
                    hc,
                    h00,
                    sdat["a0"],
                    sdat["a1"],
                    sdat["b0"],
                    sdat["b1"],
                )
            else:
                response = (fp + 1.0j * fc) * pol_phase
                fp = response.real.copy()
                fc = response.imag.copy()
                if self.lformat == "earth_time":
                    filt, norm = likelihood(
                        freqs,
                        fp,
                        fc,
                        delay,
                        dtc,
                        hp,
                        hc,
                        h00,
                        sdat["a0"],
                        sdat["a1"],
                        sdat["b0"],
                        sdat["b1"],
                    )
                else:
                    filt, norm = likelihood(
                        freqs,
                        fp,
                        fc,
                        dtc,
                        hp,
                        hc,
                        h00,
                        sdat["a0"],
                        sdat["a1"],
                        sdat["b0"],
                        sdat["b1"],
                    )
        return filt, norm, (fp, fc, dtc, hp, hc, h00)

    def calculate_hihjs(self, models):
        """Pre-calculate the hihj inner products on a grid"""
        self.hihj = {}
        self._jax_multi_likelihood_cache = {}
        for m1, m2 in itertools.combinations(models, 2):
            self.hihj[(m1, m2)] = {}
            for ifo in self.data:
                h1 = m1.h00[ifo]
                h2 = m2.h00[ifo]

                # Combine the grids
                edge = numpy.unique([m1.edges[ifo], m2.edges[ifo]])

                # Remove any points where either reference is zero
                if any(_jax_array(value) is not None for value in (h1, h2)):
                    from .relbin_jax import active_edge_bins

                    bins, fedge = active_edge_bins(h1, h2, m1.f[ifo], edge)
                else:
                    keep = numpy.where((h1[edge] != 0) | (h2[edge] != 0))[0]
                    edge = edge[keep]
                    fedge = m1.f[ifo][edge]
                    bins = numpy.array(
                        [(edge[i], edge[i + 1]) for i in range(len(edge) - 1)]
                    )
                a0, a1 = self.summary_product(h1, h2, bins, ifo)
                self.hihj[(m1, m2)][ifo] = a0, a1, fedge

    def _multi_likelihood_parts(self, m1, m2, det):
        """Evaluate one pairwise multi-signal cross term."""
        a0, a1, fedge = self.hihj[(m1, m2)][det]

        if self.still_needs_det_response:
            dtc, channel, h00 = m1._current_wf_parts[det]
            dtc2, channel2, h002 = m2._current_wf_parts[det]
            jax_data = self._get_jax_multi_likelihood_data(
                m1, m2, det, channel, channel2, fedge, h00, h002, a0, a1
            )
            if jax_data is not None:
                from .relbin_jax import (
                    likelihood_parts_det_multi as jax_mlik,
                )

                freqs_j, h00_j, h002_j, a0_j, a1_j = jax_data
                return jax_mlik(
                    freqs_j, dtc, channel, h00_j, dtc2, channel2, h002_j, a0_j, a1_j
                )

            channel = _numpy_value(channel)
            channel2 = _numpy_value(channel2)
            return self.mlik(fedge, dtc, channel, h00, dtc2, channel2, h002, a0, a1)

        fp, fc, dtc, hp, hc, h00 = m1._current_wf_parts[det]
        fp2, fc2, dtc2, hp2, hc2, h002 = m2._current_wf_parts[det]
        jax_data = self._get_jax_multi_likelihood_data(
            m1, m2, det, hp, hp2, fedge, h00, h002, a0, a1
        )
        if jax_data is not None:
            from . import relbin_jax

            freqs_j, h00_j, h002_j, a0_j, a1_j = jax_data
            if self.mlik is likelihood_parts_multi_v:
                jax_mlik = relbin_jax.likelihood_parts_multi_v
            else:
                jax_mlik = relbin_jax.likelihood_parts_multi
            return jax_mlik(
                freqs_j,
                fp,
                fc,
                dtc,
                hp,
                hc,
                h00_j,
                fp2,
                fc2,
                dtc2,
                hp2,
                hc2,
                h002_j,
                a0_j,
                a1_j,
            )

        fp, fc = _numpy_value(fp), _numpy_value(fc)
        fp2, fc2 = _numpy_value(fp2), _numpy_value(fc2)
        hp, hc = _numpy_value(hp), _numpy_value(hc)
        hp2, hc2 = _numpy_value(hp2), _numpy_value(hc2)
        return self.mlik(
            fedge, fp, fc, dtc, hp, hc, h00, fp2, fc2, dtc2, hp2, hc2, h002, a0, a1
        )

    def multi_loglikelihood(self, models):
        """Calculate a multi-model (signal) likelihood"""
        models = [self] + models
        loglr = 0
        # handle sum[<d|h_i> - 0.5 <h_i|h_i>]
        for m in models:
            loglr += m.loglr

        if not hasattr(self, "hihj"):
            self.calculate_hihjs(models)

        # Cross terms contribute
        # -0.5 * re(<h1|h2> + <h2|h1>) = -re(<h1|h2>).
        for m1, m2 in itertools.combinations(models, 2):
            for det in self.data:
                loglr -= self._multi_likelihood_parts(m1, m2, det).real
        return loglr + self.lognl

    @catch_waveform_error
    def _loglr(self):
        r"""Computes the log likelihood ratio,
        or inner product <s|h> and <h|h> if `self.return_sh_hh` is True.

        .. math::

            \log \mathcal{L}(\Theta) = \sum_i
                \left<h_i(\Theta)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta)|h_i(\Theta)\right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood ratio.
        or
        tuple
            The inner product (<s|h>, <h|h>).
        """
        # get model params
        p = self.current_params
        wfs = self.get_waveforms(p, keep_backend=True)
        lik = self.likelihood_function
        norm = 0.0
        filt = 0j
        self._current_wf_parts = {}

        for ifo in self.data:
            freqs = self.fedges[ifo]
            sdat = self.sdat[ifo]
            h00 = self.h00_sparse[ifo]

            # project waveform to detector frame if waveform does not deal
            # with detector response. Otherwise, skip detector response.

            if self.still_needs_det_response:
                dtc = 0.0

                channel = wfs[ifo]
                jax_data = None
                if lik is likelihood_parts_det:
                    jax_data = self._get_jax_likelihood_data(ifo, channel)
                if jax_data is not None:
                    from .relbin_jax import likelihood_parts_det as jax_lik

                    freqs_j, h00_j, a0_j, a1_j, b0_j, b1_j = jax_data
                    filter_i, norm_i = jax_lik(
                        freqs_j, dtc, channel, h00_j, a0_j, a1_j, b0_j, b1_j
                    )
                else:
                    channel = _numpy_value(channel)
                    filter_i, norm_i = lik(
                        freqs,
                        dtc,
                        channel,
                        h00,
                        sdat["a0"],
                        sdat["a1"],
                        sdat["b0"],
                        sdat["b1"],
                    )
                self._current_wf_parts[ifo] = (dtc, channel, h00)
            else:
                filter_i, norm_i, wf_parts = self._polarization_likelihood_parts(
                    ifo, p, wfs[ifo], lik
                )
                self._current_wf_parts[ifo] = wf_parts

            filt += filter_i
            norm += norm_i

        loglr = self.marginalize_loglr(filt, norm)
        if self.return_sh_hh:
            filt_arr = _jax_array(filt)
            if filt_arr is not None and filt_arr.ndim == 0:
                filt = filt.item()
                norm = norm.item()
            results = (filt, norm)
        else:
            results = loglr
        return results

    def max_curvature_from_reference(self):
        """Return the maximum change in slope between frequency bins
        relative to the reference waveform.
        """
        dmax = 0
        for ifo in self.data:
            waveform = self.wf_ret[ifo][0]
            waveform_arr = _jax_array(waveform)
            if waveform_arr is not None:
                import jax.numpy as jnp

                reference = self.h00_sparse[ifo]
                reference_arr = _jax_array(reference)
                if reference_arr is None:
                    reference_arr = jnp.asarray(reference)
                dtype = jnp.promote_types(waveform_arr.dtype, reference_arr.dtype)
                ratio = waveform_arr.astype(dtype) / reference_arr.astype(dtype)
                ratio = ratio / jnp.abs(ratio).min()
                curvature = ratio[2:] - 2 * ratio[1:-1] + ratio[:-2]
                d = float(jnp.abs(curvature).max())
            else:
                waveform = _numpy_value(waveform)
                r = waveform / self.h00_sparse[ifo]
                d = abs(numpy.diff(r / abs(r).min(), n=2)).max()
            dmax = d if dmax < d else dmax
        return dmax
