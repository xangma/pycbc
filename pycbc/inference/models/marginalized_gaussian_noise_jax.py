"""JAX marginalization implementations."""


import numpy


from pycbc.detector import Detector


from .gaussian_noise import catch_waveform_error


from .proposals_jax import _fused_inner_hd_hh, _inner, _jax_array, _real_inner


from .tools import marginalize_likelihood


from .marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise, MarginalizedTime, MarginalizedPolarization


from .gaussian_noise_jax import JAXBaseGaussianNoise, JAXGaussianNoise


from .proposals_jax import JAXDistMarg


from .tools_jax import whiten_template


from pycbc.types.array_jax import JAXArrayData, _divide


from pycbc.types.backend import wrap_backend_array


class JAXMarginalizedPhaseGaussianNoise(
    MarginalizedPhaseGaussianNoise, JAXGaussianNoise
):
    """JAX implementation of :class:`MarginalizedPhaseGaussianNoise`."""

    def _nowaveform_handler(self):
        """Convenience function to set loglr values if no waveform generated."""
        self._current_stats.loglikelihood = -numpy.inf
        # maxl phase doesn't exist, so set it to nan
        self._current_stats.maxl_phase = numpy.nan
        for det in self._data:
            # snr can't be < 0 by definition, so return 0
            setattr(self._current_stats, "{}_optimal_snrsq".format(det), 0.0)
        return -numpy.inf

    @catch_waveform_error
    def _loglr(self):
        r"""Computes the log likelihood ratio,
        .. math::
            \log \mathcal{L}(\Theta) =
                I_0 \left(\left|\sum_i O(h^0_i, d_i)\right|\right) -
                \frac{1}{2}\left<h^0_i, h^0_i\right>,
        at the current point in parameter space :math:`\Theta`.
        Returns
        -------
        float
            The value of the log likelihood ratio evaluated at the given point.
        """
        params = self.current_params
        if self.all_ifodata_same_rate_length:
            wfs = self.waveform_generator.generate(**params)
        else:
            wfs = {}
            for det in self.data:
                wfs.update(self.waveform_generator[det].generate(**params))
        hh = 0.0
        hd = 0j
        for det, h in wfs.items():
            # the kmax of the waveforms may be different than internal kmax
            kmax = min(len(h), self._kmax[det])
            if self._kmin[det] >= kmax:
                # if the waveform terminates before the filtering low frequency
                # cutoff, then the loglr is just 0 for this detector
                hh_i = 0.0
                hd_i = 0j
            else:
                slc = slice(self._kmin[det], kmax)
                hslc = h[slc]
                dslc = self._whitened_data[det][slc]
                wslc = self._weight[det][slc]
                if (
                    _jax_array(hslc) is not None
                    or _jax_array(dslc) is not None
                    or _jax_array(wslc) is not None
                ):
                    hd_i, hh_i = _fused_inner_hd_hh(hslc, dslc, weight=wslc)
                else:
                    hslc *= wslc
                    hh_i = hslc.inner(hslc).real
                    hd_i = hslc.inner(dslc)
            # store
            setattr(self._current_stats, "{}_optimal_snrsq".format(det), hh_i)
            hh += hh_i
            hd += hd_i
        hd_jax = _jax_array(hd)
        if hd_jax is not None:
            import jax.numpy as jnp

            self._current_stats.maxl_phase = jnp.angle(hd_jax)
        else:
            self._current_stats.maxl_phase = numpy.angle(hd)
        return marginalize_likelihood(
            hd, hh, phase=True, skip_vector=hd_jax is not None
        )

    def _batched_loglr(self, *args, **params):
        r"""Computes the phase-marginalized log likelihood ratio for a batch of
        parameter samples using ``NetworkGeometry`` and ``_fused_inner_hd_hh``.
        """
        from .gaussian_noise_jax import _batched_waveform_inner_products

        params = self._parse_batched_params(*args, **params)
        total_hd, total_hh, _, _ = _batched_waveform_inner_products(
            self, params, zero_phase=True
        )
        return marginalize_likelihood(total_hd, total_hh, phase=True, skip_vector=True)


class JAXMarginalizedTime(MarginalizedTime, JAXDistMarg, JAXBaseGaussianNoise):
    """JAX implementation of :class:`MarginalizedTime`."""

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
        """
        from pycbc.filter import matched_filter_core

        params = self.current_params
        if self.all_ifodata_same_rate_length:
            wfs = self.waveform_generator.generate(**params)
        else:
            wfs = {}
            for det in self.data:
                wfs.update(self.waveform_generator[det].generate(**params))
        sh_total = hh_total = 0.0
        snr_estimate = {}
        cplx_hpd = {}
        cplx_hcd = {}
        hphp = {}
        hchc = {}
        hphc = {}
        for det, (hp, hc) in wfs.items():
            # the kmax of the waveforms may be different than internal kmax
            kmax = min(max(len(hp), len(hc)), self._kmax[det])
            slc = slice(self._kmin[det], kmax)

            # whiten both polarizations
            hp[self._kmin[det] : kmax] = hp[slc]._return(
                JAXArrayData(whiten_template(hp[slc], self._weight[det][slc]))
            )
            hc[self._kmin[det] : kmax] = hc[slc]._return(
                JAXArrayData(whiten_template(hc[slc], self._weight[det][slc]))
            )

            # Use a higher sample rate if requested
            if self.sample_rate is not None:
                tlen = int(round(self.sample_rate * self.whitened_data[det].duration))
                flen = tlen // 2 + 1
            else:
                flen = len(self._whitened_data[det])

            hp.resize(flen)
            hc.resize(flen)
            self._whitened_data[det].resize(flen)

            cplx_hpd[det], _, _ = matched_filter_core(
                hp,
                self._whitened_data[det],
                low_frequency_cutoff=self._f_lower[det],
                high_frequency_cutoff=self._f_upper[det],
                h_norm=1,
            )
            cplx_hcd[det], _, _ = matched_filter_core(
                hc,
                self._whitened_data[det],
                low_frequency_cutoff=self._f_lower[det],
                high_frequency_cutoff=self._f_upper[det],
                h_norm=1,
            )

            hphp[det] = _real_inner(hp[slc], hp[slc])
            hchc[det] = _real_inner(hc[slc], hc[slc])
            hphc[det] = _real_inner(hp[slc], hc[slc])

            snr_proxy = (
                _normalized_snr(cplx_hpd[det], hphp[det]).squared_norm()
                + _normalized_snr(cplx_hcd[det], hchc[det]).squared_norm()
            )
            snr_estimate[det] = (0.5 * snr_proxy) ** 0.5

        self.draw_ifos(snr_estimate, log=False, **self.kwargs)
        self.snr_draw(snrs=snr_estimate)

        refframe = params.get("tc_ref_frame", "geocentric")
        ra = params["ra"]
        dec = params["dec"]
        ref_tc = params["tc"]
        pol = params["polarization"]

        for det in wfs:
            if self.precalc_antenna_factors:
                if det not in self.dets:
                    self.dets[det] = Detector(det)
                tc = self.dets[det].arrival_time(ref_tc, ra, dec, refframe)
                fp, fc, dt = self.get_precalc_antenna_factors(det)
                matched_jax = _jax_array(cplx_hpd[det])
                if matched_jax is not None:
                    from . import relbin_jax

                    pol_phase = relbin_jax.polarization_phase(pol, matched_jax)
                    fp, fc = relbin_jax.polarized_antenna_response(
                        fp, fc, pol_phase, matched_jax
                    )
                else:
                    pol_phase = numpy.exp(-2.0j * pol)
                    f = (fp + 1.0j * fc) * pol_phase
                    fp = f.real
                    fc = f.imag
            else:
                if det not in self.dets:
                    self.dets[det] = Detector(det)
                matched_jax = _jax_array(cplx_hpd[det])
                if matched_jax is not None:
                    from . import relbin_jax

                    fp, fc, tc = relbin_jax.detector_response_at_arrival(
                        self.dets[det], ref_tc, ra, dec, pol, refframe, matched_jax
                    )
                else:
                    tc = self.dets[det].arrival_time(ref_tc, ra, dec, refframe)
                    fp, fc = self.dets[det].antenna_pattern(ra, dec, pol, tc)

            cplx_hd = fp * _time_value(cplx_hpd[det], tc)
            cplx_hd += fc * _time_value(cplx_hcd[det], tc)
            hh = fp * fp * hphp[det] + fc * fc * hchc[det] + 2.0 * fp * fc * hphc[det]

            sh_total += cplx_hd
            hh_total += hh

        loglr = self.marginalize_loglr(sh_total, hh_total)
        if self.return_sh_hh:
            results = (sh_total, hh_total)
        else:
            results = loglr
        return results


class JAXMarginalizedPolarization(
    MarginalizedPolarization, JAXDistMarg, JAXBaseGaussianNoise
):
    """JAX implementation of :class:`MarginalizedPolarization`."""

    def _nowaveform_handler(self):
        """Convenience function to set loglr values if no waveform generated."""
        self._current_stats.loglr = -numpy.inf
        # maxl phase doesn't exist, so set it to nan
        self._current_stats.maxl_polarization = numpy.nan
        for det in self._data:
            # snr can't be < 0 by definition, so return 0
            setattr(self._current_stats, "{}_optimal_snrsq".format(det), 0.0)
        return -numpy.inf

    @catch_waveform_error
    def _loglr(self):
        r"""Computes the log likelihood ratio,

        .. math::

            \log \mathcal{L}(\Theta) = \sum_i
                \left<h_i(\Theta)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta)|h_i(\Theta)\right>,

        at the current parameter values :math:`\Theta`.

        Returns
        -------
        float
            The value of the log likelihood ratio.
        """
        params = self.current_params
        if self.all_ifodata_same_rate_length:
            wfs = self.waveform_generator.generate(**params)
        else:
            wfs = {}
            for det in self.data:
                wfs.update(self.waveform_generator[det].generate(**params))

        lr = sh_total = hh_total = 0.0
        refframe = params.get("tc_ref_frame", "geocentric")
        ra = params["ra"]
        dec = params["dec"]
        ref_tc = params["tc"]
        pol = params["polarization"]

        for det, (hp, hc) in wfs.items():
            if det not in self.dets:
                self.dets[det] = Detector(det)
            waveform_jax = _jax_array(hp)
            if waveform_jax is None:
                waveform_jax = _jax_array(hc)
            if waveform_jax is not None:
                from . import relbin_jax

                fp, fc, tc = relbin_jax.detector_response_at_arrival(
                    self.dets[det], ref_tc, ra, dec, pol, refframe, waveform_jax
                )
            else:
                tc = self.dets[det].arrival_time(ref_tc, ra, dec, refframe)
                fp, fc = self.dets[det].antenna_pattern(ra, dec, pol, tc)

            # the kmax of the waveforms may be different than internal kmax
            kmax = min(max(len(hp), len(hc)), self._kmax[det])
            slc = slice(self._kmin[det], kmax)

            # whiten both polarizations
            hp[self._kmin[det] : kmax] = hp[slc]._return(
                JAXArrayData(whiten_template(hp[slc], self._weight[det][slc]))
            )
            hc[self._kmin[det] : kmax] = hc[slc]._return(
                JAXArrayData(whiten_template(hc[slc], self._weight[det][slc]))
            )

            # h = fp * hp + hc * hc
            # <h, d> = fp * <hp,d> + fc * <hc,d>
            # the inner products
            cplx_hpd = _inner(hp[slc], self._whitened_data[det][slc])
            cplx_hcd = _inner(hc[slc], self._whitened_data[det][slc])

            cplx_hd = fp * cplx_hpd + fc * cplx_hcd

            # <h, h> = <fp * hp + fc * hc, fp * hp + fc * hc>
            # = Real(fpfp * <hp,hp> + fcfc * <hc,hc> + \
            #  fphc * (<hp, hc> + <hc, hp>))
            hphp = _real_inner(hp[slc], hp[slc])
            hchc = _real_inner(hc[slc], hc[slc])

            # Below could be combined, but too tired to figure out
            # if there should be a sign applied if so
            hphc = _real_inner(hp[slc], hc[slc])
            hchp = _real_inner(hc[slc], hp[slc])

            hh = fp * fp * hphp + fc * fc * hchc + fp * fc * (hphc + hchp)
            # store
            setattr(self._current_stats, "{}_optimal_snrsq".format(det), hh)
            sh_total += cplx_hd
            hh_total += hh

        lr, idx, maxl = self.marginalize_loglr(sh_total, hh_total, return_peak=True)

        # store the maxl polarization
        self._current_stats.maxl_polarization = params["polarization"][idx]
        self._current_stats.maxl_loglr = maxl

        # just store the maxl optimal snrsq
        for det in wfs:
            p = "{}_optimal_snrsq".format(det)
            setattr(self._current_stats, p, getattr(self._current_stats, p)[idx])

        return lr


def _normalized_snr(series, norm):
    """Keep scalar normalization in device storage and preserve the Series API."""
    return series._return(wrap_backend_array(_divide(_jax_array(series), norm**0.5)))


def _time_value(series, time, **kwargs):
    """Select the original interpolation independently of filtering and geometry."""
    from pycbc.types.array_jax import _reference_enabled, to_jax

    kwargs.setdefault("interpolate", "quadratic")
    if _reference_enabled("inference_time_interpolation"):
        from pycbc.reference_jax import cpu_reference

        raw = _jax_array(series)
        result = cpu_reference(
            "inference_time_interpolation",
            numpy.asarray(raw),
            spacing=series.delta_t,
            epoch=series._epoch,
            time=numpy.asarray(time)[()],
            **kwargs,
        )
        return to_jax(result, device=raw.device)
    return series.at_time(time, **kwargs)
