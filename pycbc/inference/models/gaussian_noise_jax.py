"""JAX Gaussian likelihood and explicit detector-frame batches."""

import numpy
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.waveform import FailedWaveformError, NoWaveformError
from .base_jax import JAXModelStats as ModelStats, _public_stat_value
from .tools_jax import fused_inner_hd_hh as _fused_inner_hd_hh, _jax_array

from pycbc.waveform import generator_jax as generator
from .base_jax import JAXBaseDataModel
from .gaussian_noise import BaseGaussianNoise, GaussianNoise, catch_waveform_error


class JAXBaseGaussianNoise(BaseGaussianNoise, JAXBaseDataModel):
    """JAX implementation of :class:`BaseGaussianNoise`."""

    def det_lognorm(self, det):
        """Evaluate the optional Gaussian normalization on-device."""
        if not self.normalize:
            return 0.0
        if det not in self._lognorm:
            import jax.numpy as jnp
            from pycbc.types.array_jax import _reference_enabled, _cpu_reference, to_jax

            p = _jax_array(self._psds[det])
            low, high = self._kmin[det], self._kmax[det]
            if _reference_enabled("inference_whitening"):
                total = numpy.log(numpy.asarray(p[low:high])).sum()
            else:
                values = jnp.log(p[low:high])
                total = (
                    to_jax(_cpu_reference(values, "sum"), device=p.device)
                    if _reference_enabled("sum")
                    else jnp.sum(values)
                )
            dt = self._whitened_data[det].delta_t
            constant = self._N[det] * numpy.log(numpy.pi * self._N[det] * dt) / 2.0
            self._lognorm[det] = -(constant + total)
        return self._lognorm[det]

    def update(self, **params):
        super().update(**params)
        self._current_stats = ModelStats()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from pycbc.detector import NetworkGeometry

        self.network_geometry = NetworkGeometry(list(self._data))
        self._current_stats = ModelStats()

    psds = BaseGaussianNoise.psds

    @BaseGaussianNoise.psds.setter
    def psds(self, psds):
        """Set weights on the selected device, or run the original CPU setter."""
        import jax.numpy as jnp
        from pycbc.types.array_jax import (
            JAXArrayData,
            _reference_enabled,
            _divide,
            to_jax,
        )

        if self._data is None:
            raise ValueError("No data set")
        if self._f_lower is None:
            raise ValueError("low frequency cutoff not set")
        if self._f_upper is None:
            raise ValueError("high frequency cutoff not set")
        for mapping in (
            self._psds,
            self._invpsds,
            self._weight,
            self._lognorm,
            self._det_lognls,
            self._whitened_data,
        ):
            mapping.clear()
        for det, data in self._data.items():
            raw = _jax_array(data)
            p = None if psds is None else psds[det].copy()
            low, high = self._kmin[det], self._kmax[det]
            if _reference_enabled("inference_whitening"):
                from pycbc.reference_jax import cpu_reference

                outputs = cpu_reference(
                    "inference_whitening",
                    numpy.asarray(raw),
                    spacing=data.delta_f,
                    epoch=data._epoch,
                    psd=None if p is None else p.numpy(),
                    size=self._N[det],
                    kmin=low,
                    kmax=high,
                )
                p_raw, inv, weight, whitened = (
                    to_jax(value, device=raw.device) for value in outputs
                )
            else:
                p_raw = (
                    jnp.ones(
                        int(self._N[det] / 2 + 1),
                        dtype=raw.real.dtype,
                        device=raw.device,
                    )
                    if p is None
                    else to_jax(_jax_array(p), device=raw.device)
                )
                inv = (
                    jnp.zeros_like(p_raw)
                    .at[low:high]
                    .set(_divide(1.0, p_raw[low:high]))
                )
                weight = jnp.sqrt((4 * data.delta_f) * inv)
                whitened = (raw * weight).astype(raw.dtype)
            if p is None:
                p = FrequencySeries(
                    JAXArrayData(p_raw), delta_f=data.delta_f, copy=False
                )
            else:
                p._data = JAXArrayData(p_raw)
            self._psds[det] = p
            self._invpsds[det] = FrequencySeries(
                JAXArrayData(inv), delta_f=p.delta_f, copy=False
            )
            self._weight[det] = FrequencySeries(
                JAXArrayData(weight), delta_f=p.delta_f, copy=False
            )
            self._whitened_data[det] = data._return(JAXArrayData(whitened))
        _ = self.lognl

    def _parse_batched_params(self, *args, **params):
        """Parse positional or keyword batched parameter arguments."""
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                p = dict(args[0])
                p.update(params)
                params = p
            elif len(args) == 1 and hasattr(args[0], "dtype") and args[0].dtype.names:
                p = {name: args[0][name] for name in args[0].dtype.names}
                p.update(params)
                params = p
            elif len(args) == 1 and hasattr(args[0], "_fields"):
                p = {name: getattr(args[0], name) for name in args[0]._fields}
                p.update(params)
                params = p
            else:
                raise ValueError(
                    "Unexpected positional arguments for batched evaluation"
                )
        return params

    def batched_loglikelihood(self, *args, **params):
        r"""Computes the log likelihood of a batch of parameter samples,

        .. math::

            \log p(d|\Theta_j, h) = \log \mathcal{L}(\Theta_j) + \log p(d|n),

        for each sample :math:`\Theta_j` in the batch.

        Parameters
        ----------
        *args : dict or structured array, optional
            A single positional mapping or structured array of parameter values.
        **params : dict
            Detector-frame parameter names mapped to scalars or one-
            dimensional arrays or tensors of samples. The supported batched
            parameters are ``tc``, ``ra``, ``dec``, and ``polarization``.
            Radiation-frame waveform parameters must be scalar. Batched
            evaluation requires common detector frequency grids and does not
            currently support sampling or waveform transforms, template
            recalibration, or waveform gates.
            Absolute GPS-time arrays or tensors must use 64-bit precision.

        Returns
        -------
        logl : numpy.ndarray or jax.Array
            The log likelihood values evaluated across the batch.
        """
        return self.batched_loglr(*args, **params) + self.lognl

    def batched_loglr(self, *args, **params):
        """Evaluate a batch over detector-frame extrinsic parameters.

        Batched ``tc``, ``ra``, ``dec``, and ``polarization`` values are
        supported. Radiation-frame waveform parameters must be scalar.
        Sampling and waveform transforms are not supported. A waveform
        generation failure returns one negative infinity per sample, subject
        to the same ``ignore_failed_waveforms`` policy as scalar evaluation.
        """
        params = self._parse_batched_params(*args, **params)
        try:
            return self._batched_loglr(**params)
        except NoWaveformError:
            pass
        except NotImplementedError:
            raise
        except (RuntimeError, FailedWaveformError):
            if not self.ignore_failed_waveforms:
                raise

        full_params = dict(self.static_params)
        full_params.update(params)
        generator = self.waveform_generator
        batch_size = (
            _detector_frame_batch_size(generator, params, full_params)
            if hasattr(generator, "location_args")
            else 1
        )
        # Prefer the data's device, as for the ordinary likelihood, and keep
        # the scalar model state untouched by this independent batch call.
        for value in (*self._data.values(), *full_params.values()):
            arr = _jax_array(value)
            if arr is not None:
                import jax.numpy as jnp

                dtype = arr.real.dtype
                if not jnp.issubdtype(dtype, jnp.floating):
                    dtype = jnp.float64
                return jnp.full((batch_size,), -numpy.inf, dtype=dtype)
        return numpy.full(batch_size, -numpy.inf)

    def _batched_loglr(self, *args, **params):
        """Computes the log likelihood ratio for a batch of parameter samples.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Batched loglr not implemented for this model.")


class JAXGaussianNoise(GaussianNoise, JAXBaseGaussianNoise):
    """JAX implementation of :class:`GaussianNoise`."""

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
        float or jax.Array
            The scalar log likelihood ratio.
        """
        wfs = self.get_waveforms()
        lr = 0.0
        for det, h in wfs.items():
            # the kmax of the waveforms may be different than internal kmax
            kmax = min(len(h), self._kmax[det])
            if self._kmin[det] >= kmax:
                # if the waveform terminates before the filtering low frequency
                # cutoff, then the loglr is just 0 for this detector
                cplx_hd = 0j
                hh = 0.0
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
                    cplx_hd, hh = _fused_inner_hd_hh(hslc, dslc, weight=wslc)
                else:
                    # Preserve the native CPU scalar algorithm and its
                    # established weighting/mutation semantics. The fused
                    # helper is intended for backend and batched paths.
                    hslc *= wslc
                    cplx_hd = hslc.inner(dslc)
                    hh = hslc.inner(hslc).real
            cplx_loglr = cplx_hd - 0.5 * hh
            # store
            setattr(self._current_stats, "{}_optimal_snrsq".format(det), hh)
            setattr(self._current_stats, "{}_cplx_loglr".format(det), cplx_loglr)
            lr += cplx_loglr.real
        # also store the loglikelihood, to ensure it is populated in the
        # current stats even if loglikelihood is never called
        self._current_stats.loglikelihood = lr + self.lognl
        return float(lr) if numpy.isscalar(lr) else lr

    def det_cplx_loglr(self, det):
        """Returns the complex log likelihood ratio in the given detector.

        Parameters
        ----------
        det : str
            The name of the detector.

        Returns
        -------
        complex float :
            The complex log likelihood ratio.
        """
        # try to get it from current stats
        try:
            value = getattr(self._current_stats, "{}_cplx_loglr".format(det))
        except AttributeError:
            # Populate through the cached public property. Calling ``_loglr``
            # directly would leave ``current_stats.loglr`` unset, so a later
            # ``self.loglr`` access could regenerate and re-weight waveforms.
            _ = self.loglr
            # now try returning again
            value = getattr(self._current_stats, "{}_cplx_loglr".format(det))
        return _public_stat_value(value)

    def det_optimal_snrsq(self, det):
        """Returns the opitmal SNR squared in the given detector.

        Parameters
        ----------
        det : str
            The name of the detector.

        Returns
        -------
        float :
            The opimtal SNR squared.
        """
        # try to get it from current stats
        try:
            value = getattr(self._current_stats, "{}_optimal_snrsq".format(det))
        except AttributeError:
            # Populate through the cached public property; see the complex
            # detector-stat accessor above.
            _ = self.loglr
            # now try returning again
            value = getattr(self._current_stats, "{}_optimal_snrsq".format(det))
        return _public_stat_value(value)

    def _batched_loglr(self, *args, **params):
        r"""Computes the log likelihood ratio for a batch of parameter samples,

        .. math::

            \log \mathcal{L}(\Theta_j) = \sum_i \left[
                \left<h_i(\Theta_j)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta_j)|h_i(\Theta_j)\right> \right],

        simultaneously for all samples :math:`j` using ``NetworkGeometry`` and
        fused inner products ``_fused_inner_hd_hh``.

        Parameters
        ----------
        *args : dict or structured array, optional
            A single positional mapping or structured array of parameter values.
        **params : dict
            Detector-frame parameter names mapped to scalars or one-
            dimensional arrays or tensors of samples. Radiation-frame
            waveform parameters must be scalar. Batched evaluation requires
            common detector frequency grids and does not currently support
            template recalibration or waveform gates. Absolute GPS-time arrays
            or tensors must use 64-bit precision.

        Returns
        -------
        loglr : numpy.ndarray or jax.Array
            The log likelihood ratio values evaluated across the batch.
        """
        params = self._parse_batched_params(*args, **params)
        total_hd, total_hh, _, _ = _batched_waveform_inner_products(
            self, params, zero_phase=False
        )
        return total_hd.real - 0.5 * total_hh


def _batched_value_shape(value):
    """Return a parameter value's shape without moving backend data to host."""
    arr = _jax_array(value)
    if arr is not None:
        return tuple(arr.shape)
    return numpy.shape(value)


def _detector_frame_batch_size(generator, supplied_params, full_params):
    """Validate detector-frame-only batching and return its sample count."""
    location_args = set(generator.location_args)
    unsupported = sorted(
        name
        for name, value in supplied_params.items()
        if name not in location_args and _batched_value_shape(value)
    )
    if unsupported:
        raise ValueError(
            "Batched radiation-frame waveform parameters are not supported; "
            "only tc, ra, dec, and polarization may be batched. Got: "
            + ", ".join(unsupported)
        )

    lengths = {}
    for name in location_args:
        if name not in full_params:
            continue
        shape = _batched_value_shape(full_params[name])
        if len(shape) > 1:
            raise ValueError(
                f"Batched detector-frame parameter {name!r} must be scalar "
                "or one-dimensional"
            )
        if shape:
            lengths[name] = shape[0]

    batch_size = max(lengths.values(), default=1)
    if batch_size < 1:
        raise ValueError("Batched detector-frame parameters cannot be empty")
    mismatched = {
        name: length
        for name, length in lengths.items()
        if length not in (1, batch_size)
    }
    if mismatched:
        detail = ", ".join(
            f"{name}={length}" for name, length in sorted(mismatched.items())
        )
        raise ValueError(
            "Batched detector-frame parameter lengths must be one or the "
            f"common batch length {batch_size}; got {detail}"
        )
    return batch_size


def _batched_waveform_inner_products(model, params, zero_phase=False):
    """Evaluate batched waveform frequency-domain projections and inner products
    across all detectors in a single pass using NetworkGeometry and
    _fused_inner_hd_hh.
    """
    if getattr(model, "sampling_transforms", None) is not None or getattr(
        model, "waveform_transforms", None
    ):
        raise NotImplementedError(
            "Batched likelihood evaluation does not support sampling or "
            "waveform transforms; use scalar model evaluation"
        )
    full_params = dict(model.static_params) if model.static_params else {}
    full_params.update(params)

    gen = model.waveform_generator if hasattr(model, "waveform_generator") else None
    if not getattr(model, "all_ifodata_same_rate_length", True):
        raise NotImplementedError(
            "Batched likelihood evaluation requires all detectors to use "
            "the same sample rate and segment length"
        )
    if isinstance(gen, dict):
        raise NotImplementedError(
            "Batched likelihood evaluation does not support per-detector "
            "waveform generators"
        )

    recalibration = getattr(model, "recalibration", None)
    if not recalibration and gen is not None:
        recalibration = getattr(gen, "recalib", None)
    gates = getattr(model, "gates", None)
    if not gates and gen is not None:
        gates = getattr(gen, "gates", None)
    if recalibration:
        raise NotImplementedError(
            "Batched likelihood evaluation does not support template recalibration"
        )
    if gates:
        raise NotImplementedError(
            "Batched likelihood evaluation does not support waveform gates"
        )

    has_rframe = (
        gen is not None
        and hasattr(gen, "rframe_generator")
        and hasattr(gen, "location_args")
    )

    if not has_rframe:
        shaped_params = sorted(
            name for name, value in params.items() if _batched_value_shape(value)
        )
        if shaped_params:
            raise NotImplementedError(
                "Batched likelihood parameters require a radiation-frame "
                "waveform generator with an explicit detector-frame "
                "contract. Got: " + ", ".join(shaped_params)
            )

    if has_rframe:
        batch_size = _detector_frame_batch_size(gen, params, full_params)
        rfparams = {
            param: full_params[param]
            for param in full_params
            if param not in gen.location_args
        }
        if zero_phase:
            rfparams["coa_phase"] = 0.0
        hp, hc = gen.rframe_generator.generate(**rfparams)
        from pycbc.types.array_jax import to_jax
        from pycbc.types.backend import wrap_backend_array

        if _jax_array(hp) is None:
            hp = hp._return(wrap_backend_array(to_jax(numpy.asarray(hp))))
        if _jax_array(hc) is None:
            hc = hc._return(wrap_backend_array(to_jax(numpy.asarray(hc))))
        if isinstance(hp, TimeSeries):
            df = full_params.get(
                "delta_f", gen.current_params.get("delta_f", hp.delta_f)
            )
            hp = hp.to_frequencyseries(delta_f=df)
            hc = hc.to_frequencyseries(delta_f=df)
            tshift = 1.0 / df - abs(hp._epoch)
        else:
            tshift = 0.0

        epoch = getattr(gen, "_epoch", float(getattr(hp, "_epoch", 0.0)))
        delta_f = hp.delta_f

        ra = full_params.get("ra", 0.0)
        dec = full_params.get("dec", 0.0)
        pol = full_params.get("polarization", 0.0)
        ref_tc = full_params.get("tc", 0.0)
        refframe = full_params.get("tc_ref_frame", "geocentric")
        generator._validate_absolute_time_precision(ref_tc, epoch)

        total_hd = None
        total_hh = None
        det_hd = {}
        det_hh = {}
        delay_dict = None
        if (
            refframe == "geocentric"
            and getattr(model, "network_geometry", None) is not None
        ):
            delays = model.network_geometry.time_delay_from_earth_center(
                ra, dec, ref_tc
            )
            delay_dict = model.network_geometry.to_dict(delays)

        for detname in model._data:
            if getattr(model, "network_geometry", None) is not None:
                det = model.network_geometry[detname]
            else:
                from pycbc.detector import Detector

                det = Detector(detname)
            offset = (
                delay_dict[detname]
                if delay_dict is not None
                else generator._detector_time_offset(det, ref_tc, ra, dec, refframe)
            )
            tc, dt = generator._arrival_time_and_shift(ref_tc, offset, epoch, tshift)
            fp, fc = det.antenna_pattern(ra, dec, pol, tc)

            from .tools_jax import batched_project_detector_strain

            h_det = batched_project_detector_strain(
                hp, hc, fp, fc, dt, delta_f, batch_size
            )

            kmax = min(h_det.shape[-1], model._kmax[detname])
            kmin = model._kmin[detname]
            if kmin >= kmax:
                import jax.numpy as jnp

                cplx_hd = jnp.zeros(h_det.shape[:-1], dtype=h_det.dtype)
                hh = jnp.zeros(h_det.shape[:-1], dtype=h_det.real.dtype)
            else:
                slc = slice(kmin, kmax)
                w_slc = model._weight[detname][slc]
                d_slc = model._whitened_data[detname][slc]
                cplx_hd, hh = _fused_inner_hd_hh(h_det[..., slc], d_slc, weight=w_slc)

            det_hd[detname] = cplx_hd
            det_hh[detname] = hh
            total_hd = cplx_hd if total_hd is None else total_hd + cplx_hd
            total_hh = hh if total_hh is None else total_hh + hh

        return total_hd, total_hh, det_hd, det_hh
    else:
        if gen is not None:
            wfs = gen.generate(**full_params)
        else:
            wfs = model.get_waveforms()

        total_hd = None
        total_hh = None
        det_hd = {}
        det_hh = {}
        for detname, h in wfs.items():
            kmax = min(
                h.shape[-1] if hasattr(h, "shape") else len(h), model._kmax[detname]
            )
            kmin = model._kmin[detname]
            if kmin >= kmax:
                cplx_hd = 0j
                hh = 0.0
            else:
                slc = slice(kmin, kmax)
                w_slc = model._weight[detname][slc]
                d_slc = model._whitened_data[detname][slc]
                cplx_hd, hh = _fused_inner_hd_hh(h[..., slc], d_slc, weight=w_slc)
            det_hd[detname] = cplx_hd
            det_hh[detname] = hh
            total_hd = cplx_hd if total_hd is None else total_hd + cplx_hd
            total_hh = hh if total_hh is None else total_hh + hh
        return total_hd, total_hh, det_hd, det_hh
