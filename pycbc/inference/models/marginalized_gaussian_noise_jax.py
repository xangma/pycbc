"""JAX marginalization implementations."""


import numpy


from .gaussian_noise import catch_waveform_error


from .proposals_jax import _fused_inner_hd_hh, _jax_array


from .tools import marginalize_likelihood


from .marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise


from .gaussian_noise_jax import JAXGaussianNoise


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
