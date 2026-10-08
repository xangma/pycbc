"""JAX marginalization implementations."""

import itertools
import numpy
from scipy import special
from pycbc.detector import Detector
from pycbc.waveform import generator
from .gaussian_noise import catch_waveform_error, create_waveform_generator
from .proposals_jax import _fused_inner_hd_hh, _inner, _jax_array, _real_inner
from .tools import marginalize_likelihood

from .marginalized_gaussian_noise import (
    MarginalizedPhaseGaussianNoise,
    MarginalizedTime,
    MarginalizedPolarization,
    MarginalizedHMPolPhase,
)
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


class JAXMarginalizedHMPolPhase(MarginalizedHMPolPhase, JAXBaseGaussianNoise):
    """JAX implementation of :class:`MarginalizedHMPolPhase`."""

    def __init__(
        self,
        variable_params,
        data,
        low_frequency_cutoff,
        psds=None,
        high_frequency_cutoff=None,
        normalize=False,
        polarization_samples=100,
        coa_phase_samples=100,
        static_params=None,
        **kwargs,
    ):
        # set up the boiler-plate attributes
        super(MarginalizedHMPolPhase, self).__init__(
            variable_params,
            data,
            low_frequency_cutoff,
            psds=psds,
            high_frequency_cutoff=high_frequency_cutoff,
            normalize=normalize,
            static_params=static_params,
            **kwargs,
        )
        # create the waveform generator
        self.waveform_generator = create_waveform_generator(
            self.variable_params,
            self.data,
            waveform_transforms=self.waveform_transforms,
            recalibration=self.recalibration,
            generator_class=generator.FDomainDetFrameModesGenerator,
            gates=self.gates,
            **self.static_params,
        )
        pol = numpy.linspace(0, 2 * numpy.pi, polarization_samples)
        phase = numpy.linspace(0, 2 * numpy.pi, coa_phase_samples)
        # remap to every combination of the parameters
        # this gets every combination by mappin them to an NxM grid
        # one needs to be transposed so that they run allong opposite
        # dimensions
        n = coa_phase_samples * polarization_samples
        self.nsamples = n
        self.pol = numpy.resize(pol, n)
        phase = numpy.resize(phase, n)
        phase = phase.reshape(coa_phase_samples, polarization_samples)
        self.phase = phase.T.flatten()
        self._phase_fac = {}
        self._jax_marginalization_grids = {}
        self._jax_phase_fac = {}
        self.dets = {}

    def _marginalization_grids(self, like):
        """Return fixed polarization and phase grids on ``like``'s device."""
        jax_arr = _jax_array(like)
        if jax_arr is not None:
            import jax.numpy as jnp

            key = (jax_arr.dtype, jax_arr.device)
            try:
                return self._jax_marginalization_grids[key]
            except KeyError:
                grids = tuple(
                    jnp.asarray(
                        values, dtype=jnp.real(jax_arr).dtype, device=jax_arr.device
                    )
                    for values in (self.pol, self.phase)
                )
                self._jax_marginalization_grids[key] = grids
                return grids

        return self.pol, self.phase

    def phase_fac(self, m, phase=None):
        r"""The phase :math:`\exp[i m \phi]`."""
        use_default_phase = phase is None
        if use_default_phase:
            phase = self.phase
        phase_jax = _jax_array(phase)
        if phase_jax is not None:
            import jax.numpy as jnp

            angle = m * phase_jax
            factor = jnp.cos(angle) + 1j * jnp.sin(angle)
            is_fixed_grid = any(
                phase_jax is grids[1]
                for grids in self._jax_marginalization_grids.values()
            )
            if not is_fixed_grid:
                return factor
            key = (m, phase_jax.dtype, phase_jax.device)
            try:
                return self._jax_phase_fac[key]
            except KeyError:
                self._jax_phase_fac[key] = factor
                return factor

        if not use_default_phase:
            return numpy.exp(1.0j * m * numpy.asarray(phase))
        try:
            return self._phase_fac[m]
        except KeyError:
            # hasn't been computed yet, calculate it
            self._phase_fac[m] = numpy.exp(1.0j * m * self.phase)
            return self._phase_fac[m]

    def _nowaveform_handler(self):
        """Convenience function to set loglr values if no waveform generated."""
        # maxl phase doesn't exist, so set it to nan
        self._current_stats.maxl_polarization = numpy.nan
        self._current_stats.maxl_phase = numpy.nan
        return -numpy.inf

    @catch_waveform_error
    def _loglr(self, return_unmarginalized=False):
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
        wfs = self.waveform_generator.generate(**params)

        # ---------------------------------------------------------------------
        # Some optimizations not yet taken:
        # * higher m calculations could have a lot of redundancy
        # * fp/fc need not be calculated except where polarization is different
        # * may be possible to simplify this by making smarter use of real/imag
        # ---------------------------------------------------------------------
        lr = 0.0
        hds = {}
        hhs = {}
        refframe = params.get("tc_ref_frame", "geocentric")
        ra = params["ra"]
        dec = params["dec"]
        ref_tc = params["tc"]
        first_mode = next(iter(next(iter(wfs.values())).values()), (None, None))[0]
        pol, phase = self._marginalization_grids(first_mode)

        if (
            refframe == "geocentric"
            and getattr(self, "network_geometry", None) is not None
        ):
            delay_net = self.network_geometry.time_delay_from_earth_center(
                ra, dec, ref_tc
            )
            delay_dict = self.network_geometry.to_dict(delay_net)
        else:
            delay_dict = {}

        for det, modes in wfs.items():
            if det in delay_dict:
                tc = ref_tc + delay_dict[det]
                detector = self.network_geometry[det]
                fp, fc = detector.antenna_pattern(ra, dec, pol, tc)
            else:
                if det not in self.dets:
                    self.dets[det] = Detector(det)
                waveform_jax = _jax_array(first_mode)
                if waveform_jax is not None:
                    from . import relbin_jax

                    fp, fc, tc = relbin_jax.detector_response_at_arrival(
                        self.dets[det], ref_tc, ra, dec, pol, refframe, waveform_jax
                    )
                else:
                    tc = self.dets[det].arrival_time(ref_tc, ra, dec, refframe)
                    fp, fc = self.dets[det].antenna_pattern(ra, dec, pol, tc)

            # loop over modes and prepare the waveform modes
            # we will sum up zetalm = glm <ulm, d> + i glm <vlm, d>
            # over all common m so that we can apply the phase once
            zetas = {}
            rlms = {}
            slms = {}
            for mode in modes:
                l, m = mode
                ulm, vlm = modes[mode]

                # whiten the waveforms
                # the kmax of the waveforms may be different than internal kmax
                kmax = min(max(len(ulm), len(vlm)), self._kmax[det])
                slc = slice(self._kmin[det], kmax)
                ulm[self._kmin[det] : kmax] = ulm[slc]._return(
                    JAXArrayData(whiten_template(ulm[slc], self._weight[det][slc]))
                )
                vlm[self._kmin[det] : kmax] = vlm[slc]._return(
                    JAXArrayData(whiten_template(vlm[slc], self._weight[det][slc]))
                )

                # the inner products
                # <ulm, d>
                ulmd = _real_inner(ulm[slc], self._whitened_data[det][slc])
                # <vlm, d>
                vlmd = _real_inner(vlm[slc], self._whitened_data[det][slc])

                # add inclination, and pack into a complex number
                import lal

                glm = lal.SpinWeightedSphericalHarmonic(
                    params["inclination"], 0, -2, l, m
                ).real

                if m not in zetas:
                    zetas[m] = 0j
                zetas[m] += glm * (ulmd + 1j * vlmd)

                # Get condense set of the parts of the waveform that only diff
                # by m, this is used next to help calculate <h, h>
                r = glm * ulm
                s = glm * vlm

                if m not in rlms:
                    rlms[m] = r
                    slms[m] = s
                else:
                    rlms[m] += r
                    slms[m] += s

            # now compute all possible <hlm, hlm>
            rr_m = {}
            ss_m = {}
            rs_m = {}
            sr_m = {}
            combos = itertools.combinations_with_replacement(rlms.keys(), 2)
            for m, mprime in combos:
                r = rlms[m]
                s = slms[m]
                rprime = rlms[mprime]
                sprime = slms[mprime]
                rr_m[mprime, m] = _real_inner(r[slc], rprime[slc])
                ss_m[mprime, m] = _real_inner(s[slc], sprime[slc])
                rs_m[mprime, m] = _real_inner(s[slc], rprime[slc])
                sr_m[mprime, m] = _real_inner(r[slc], sprime[slc])
                # store the conjugate for easy retrieval later
                rr_m[m, mprime] = rr_m[mprime, m]
                ss_m[m, mprime] = ss_m[mprime, m]
                rs_m[m, mprime] = sr_m[mprime, m]
                sr_m[m, mprime] = rs_m[mprime, m]
            # now apply the phase to all the common ms
            hpd = 0.0
            hcd = 0.0
            hphp = 0.0
            hchc = 0.0
            hphc = 0.0
            for m, zeta in zetas.items():
                phase_coeff = (
                    self.phase_fac(m)
                    if phase is self.phase
                    else self.phase_fac(m, phase)
                )

                # <h+, d> = (exp[i m phi] * zeta).real()
                # <hx, d> = -(exp[i m phi] * zeta).imag()
                cosm = phase_coeff.real
                sinm = phase_coeff.imag
                if _jax_array(phase_coeff) is not None:
                    hpd += cosm * zeta.real - sinm * zeta.imag
                    hcd -= cosm * zeta.imag + sinm * zeta.real
                else:
                    z = phase_coeff * zeta
                    hpd += z.real
                    hcd -= z.imag

                for mprime in zetas:
                    pcprime = (
                        self.phase_fac(mprime)
                        if phase is self.phase
                        else self.phase_fac(mprime, phase)
                    )

                    cosmprime = pcprime.real
                    sinmprime = pcprime.imag
                    # needed components
                    rr = rr_m[m, mprime]
                    ss = ss_m[m, mprime]
                    rs = rs_m[m, mprime]
                    sr = sr_m[m, mprime]
                    # <hp, hp>
                    hphp += (
                        rr * cosm * cosmprime
                        + ss * sinm * sinmprime
                        - rs * cosm * sinmprime
                        - sr * sinm * cosmprime
                    )
                    # <hc, hc>
                    hchc += (
                        rr * sinm * sinmprime
                        + ss * cosm * cosmprime
                        + rs * sinm * cosmprime
                        + sr * cosm * sinmprime
                    )
                    # <hp, hc>
                    hphc += (
                        -rr * cosm * sinmprime
                        + ss * sinm * cosmprime
                        + sr * sinm * sinmprime
                        - rs * cosm * cosmprime
                    )

            # Now apply the polarizations and calculate the loglr
            # We have h = Fp * hp + Fc * hc
            # loglr = <h, d> - <h, h>/2
            #       = Fp*<hp, d> + Fc*<hc, d>
            #          - (1/2)*(Fp*Fp*<hp, hp> + Fc*Fc*<hc, hc>
            #                   + 2*Fp*Fc<hp, hc>)
            # (in the last line we have made use of the time series being
            #  real, so that <a, b> = <b, a>).
            hd = fp * hpd + fc * hcd
            hh = fp * fp * hphp + fc * fc * hchc + 2 * fp * fc * hphc
            hds[det] = hd
            hhs[det] = hh
            lr += hd - 0.5 * hh

        if return_unmarginalized:
            return self.pol, self.phase, lr, hds, hhs

        lr_jax = _jax_array(lr)
        if lr_jax is not None:
            import jax.numpy as jnp
            import jax.scipy.special as jsp

            from pycbc.types.array_jax import _reference_enabled

            if _reference_enabled("inference_marginalization"):
                lr_total = special.logsumexp(numpy.asarray(lr_jax), axis=0) - numpy.log(
                    self.nsamples
                )
            else:
                lr_total = jsp.logsumexp(lr_jax, axis=0) - jnp.log(self.nsamples)
            idx = int(jnp.argmax(lr_jax))
            lr_total = float(lr_total)
        else:
            lr_total = special.logsumexp(lr) - numpy.log(self.nsamples)
            idx = lr.argmax()

        # store the maxl values
        self._current_stats.maxl_polarization = self.pol[idx]
        self._current_stats.maxl_phase = self.phase[idx]
        return float(lr_total)


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
