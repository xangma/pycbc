"""Device-resident proposal and marginalization support."""

import logging
import numpy
import numpy.random
import tqdm
from scipy.special import logsumexp
from pycbc.detector import Detector
from pycbc.types.backend import backend_array, is_backend

from .tools import DistMarg

EARTH_RADIUS = 0.031


def _jax_array(value):
    """Return the public JAX backend array, if present."""
    return backend_array(value, "jax")


def _is_backend_array(value):
    """Return whether value is backed by JAX."""
    return is_backend(value, "jax")


def _jax_tools(*values):
    """Load the JAX implementation only for JAX-backed inputs."""
    if not any(is_backend(value, "jax") for value in values):
        return None
    from pycbc.inference.models import tools_jax

    return tools_jax


def _backend_tools(*values):
    """Load the matching backend module (JAX) if present."""
    return _jax_tools(*values)


def _inner(left, right):
    """Return an inner product without scalarizing backend reductions."""
    backend = _backend_tools(left, right)
    if backend is not None and all(is_backend(value, "jax") for value in (left, right)):
        return backend.inner(left, right)
    return left.inner(right)


def _real_inner(left, right):
    """Return a real inner product without scalarizing backend reductions."""
    backend = _backend_tools(left, right)
    if backend is not None and all(is_backend(value, "jax") for value in (left, right)):
        return backend.real_inner(left, right)
    return _inner(left, right).real


def _fused_inner_hd_hh(h, d, weight=None):
    """Use the JAX implementation without adding a separate CPU algorithm."""
    from .tools_jax import fused_inner_hd_hh

    return fused_inner_hd_hh(h, d, weight)


def _squared_norm_values(snr):
    """Return SNR-squared values without moving backend series to the host."""
    values = snr.squared_norm()
    arr = _jax_array(values)
    if arr is not None:
        return arr
    return values.numpy()


def _selected_values(values, indices, *, host=True):
    """Return selected proposal values, optionally preserving backend storage."""
    backend = _backend_tools(values)
    if backend is not None:
        return backend.selected_values(values, indices, host=host)
    return values[indices]


def _add_values(total, values):
    """Add proposal values without moving a backend operand to the host."""
    if total is None:
        return values

    backend = _backend_tools(total, values)
    if backend is not None:
        return backend.add_values(total, values)
    return total + values


def _last_index_at_or_below(values, upper):
    """Return the last sorted-value index at or below ``upper``."""
    backend = _backend_tools(values)
    if backend is not None:
        return backend.last_index_at_or_below(values, upper)
    if hasattr(values, "numpy"):
        values = values.numpy()
    insertion = numpy.searchsorted(values, upper, side="right")

    if insertion == 0:
        raise IndexError(f"no values are at or below {upper}")
    return int(insertion - 1)


def _threshold_extent(values, threshold):
    """Return the first and last indices above ``threshold``."""
    backend = _backend_tools(values)
    if backend is not None:
        return backend.threshold_extent(values, threshold)
    indices = numpy.flatnonzero(abs(values) > threshold)
    return int(indices[0]), int(indices[-1])


def draw_sample(loglr, size=None, *, host=True):
    """Draw a random index from a 1-d vector with loglr weights.

    Vector draws from backend inputs may be kept on their input device by
    setting ``host=False``. Every path uses the caller's NumPy random stream;
    host-returning and scalar draws retain the original public return types.
    """
    backend = _backend_tools(loglr)
    if backend is not None:
        return backend.draw_sample(loglr, size=size, host=host)

    if size:
        x = numpy.random.uniform(size=size)
    else:
        x = numpy.random.uniform()
    loglr = loglr - loglr.max()
    cdf = numpy.exp(loglr).cumsum()
    cdf /= cdf[-1]
    xl = numpy.searchsorted(cdf, x)
    return xl


def _draw_device_sample_with_host_rng(loglr, size):
    """Draw device indices while retaining NumPy's RNG stream.

    Sky/time marginalization historically uses NumPy to generate its random
    uniforms even when the proposal weights live on a device. Keeping
    the uniforms on that stream preserves seeded public results while the
    returned indices can remain device-resident for subsequent gathers.
    """
    backend = _backend_tools(loglr)
    if backend is None:
        raise TypeError("device sampling requires backend-backed weights")
    return backend.draw_device_sample_with_host_rng(loglr, size)


def _device_index_matrix_to_host(indices):
    """Materialize device index vectors at one explicit host boundary."""
    if is_backend(indices[0], "jax"):
        from pycbc.inference.models import tools_jax

        return tools_jax.device_index_matrix_to_host(indices)
    return numpy.asarray([numpy.asarray(i) for i in indices], dtype=numpy.int64)


def _draw_sky_time_indices(weights, offsets, size):
    """Draw detector indices and return their host-relative delays.

    Same-device backend weights keep each draw on device through proposal
    gathers and integer detector-offset arithmetic. The complete index matrix
    then crosses to the host once for the existing sky-delay dictionary and
    GPS-time construction. NumPy, mixed-backend, and mixed-device inputs keep
    the legacy host-index behavior.
    """
    if len(weights) != len(offsets) or not weights:
        raise ValueError("weights and offsets must have the same nonzero size")

    backend = _backend_tools(*weights)
    use_device_indices = (
        bool(size)
        and backend is not None
        and all(is_backend(weight, "jax") for weight in weights)
        and backend.same_device(weights)
    )

    indices = []
    selected_weights = []
    for weight, offset in zip(weights, offsets, strict=True):
        if use_device_indices:
            index = _draw_device_sample_with_host_rng(weight, size)
        else:
            index = draw_sample(weight, size=size)
        selected_weights.append(_selected_values(weight, index, host=False))
        indices.append(index + offset)

    if use_device_indices:
        host_indices = _device_index_matrix_to_host(indices)
    else:
        host_indices = numpy.stack(indices, axis=0)

    reference = host_indices[0]
    relative = [reference - index for index in host_indices[1:]]
    return reference, relative, selected_weights


def _weighted_loglr(values, weights):
    """Add probability weights without moving backend values to the host."""
    backend = _backend_tools(values)
    if backend is not None:
        return backend.weighted_loglr(values, weights)
    return values + numpy.log(weights)


def _phase_reconstruction_values(sh, hh, sample_count=int(1e4)):
    """Build the phase-reconstruction likelihood grid on its input device."""
    backend = _backend_tools(sh, hh)
    if backend is not None:
        return backend.phase_reconstruction_values(sh, hh, sample_count=sample_count)
    phase = numpy.linspace(0, numpy.pi * 2.0, sample_count)
    loglr = (numpy.exp(-2.0j * phase) * sh).real + hh
    return phase, loglr


def _selected_scalar(values, index):
    """Return one selected value at the public scalar boundary."""
    backend = _backend_tools(values)
    if backend is not None:
        return backend.selected_scalar(values, index)
    return values[index]


def _random_permutation(values, size, rng):
    """Return backend and host indices for sampling without replacement."""
    backend = _backend_tools(values)
    if backend is not None:
        return backend.random_permutation(values, size, rng)
    choice = rng.choice(len(values), size=size, replace=False)
    return choice, choice


def _normalize_logweights(values):
    """Normalize log weights using their native reduction backend."""
    backend = _backend_tools(values)
    if backend is not None:
        return backend.normalize_logweights(values)
    return values - logsumexp(values)


def _host_indices(values):
    """Materialize indices only when they are device-backed."""
    backend = _backend_tools(values)
    if backend is not None:
        return backend.host_indices(values)
    return values


def setup_distance_marg_interpolant(*args, **kwargs):
    """Build the established static grid, then evaluate it on-device."""
    from .tools import setup_distance_marg_interpolant as original

    return original(*args, **kwargs)


def _numpy_from_backend(value):
    """Return a detached CPU value for a NumPy-only calculation."""
    backend = _backend_tools(value)
    if backend is None:
        return value
    return backend.numpy_from_backend(value)


def _marginalize_likelihood_jax(
    sh,
    hh,
    logw,
    phase,
    distance,
    skip_vector,
    return_peak,
    return_complex,
    interpolator=None,
):
    """JAX implementation of explicit likelihood marginalizations."""
    backend = _jax_tools(sh, hh)
    return backend.marginalize_likelihood(
        sh,
        hh,
        logw,
        phase,
        distance,
        skip_vector,
        return_peak,
        return_complex,
        interpolator=interpolator,
    )


class JAXDistMarg(DistMarg):
    """JAX implementation of :class:`DistMarg`."""

    def premarg_draw(self):
        """Choose random samples from a precomputed proposal set.

        The model keeps its established RNG stream. JAX proposal weights
        remain on the device for selection and normalization.
        """

        # Update the current proposed times and the marginalization values
        logw = self.premarg["logw_partial"]
        if self.vsamples == len(logw):
            choice = slice(None, None)
            host_choice = choice
        else:
            choice, host_choice = _random_permutation(
                logw, self.vsamples, self._choice_rng
            )

        for k in self.snr_params:
            values = self.premarg[k]
            indices = choice if _is_backend_array(values) else host_choice
            self.marginalize_vector_params[k] = _selected_values(
                values, indices, host=False
            )

        self._current_params.update(self.marginalize_vector_params)
        sample_idx = self.premarg["sample_idx"]
        indices = choice if _is_backend_array(sample_idx) else host_choice
        self.sample_idx = _selected_values(sample_idx, indices, host=False)

        # Update the importance weights for each vector sample
        selected_logw = _selected_values(logw, choice, host=False)
        logw = _add_values(self.marginalize_vector_weights, selected_logw)
        self.marginalize_vector_weights = _normalize_logweights(logw)
        return self.marginalize_vector_params

    def draw_times(self, snrs, size=None):
        """Draw times consistent with the incoherent network SNR

        Parameters
        ----------
        snrs: dist of TimeSeries
        """
        if not hasattr(self, "tinfo"):
            # determine the rough time offsets for this sky location
            tcprior = self.marginalized_vector_priors["tc"]
            tcmin, tcmax = tcprior.bounds["tc"]
            tcave = (tcmax + tcmin) / 2.0
            ifos = list(snrs.keys())
            if hasattr(self, "keep_ifos"):
                ifos = self.keep_ifos
            d = {ifo: Detector(ifo, reference_time=tcave) for ifo in ifos}
            self.tinfo = tcmin, tcmax, tcave, ifos, d
            self.snr_params = ["tc"]

        tcmin, tcmax, tcave, ifos, d = self.tinfo
        vsamples = size if size is not None else self.vsamples

        # Determine the weights for the valid time range
        ra = self._current_params["ra"]
        dec = self._current_params["dec"]

        # Determine the common valid time range
        iref = ifos[0]
        dref = d[iref]
        dt = dref.time_delay_from_earth_center(ra, dec, tcave)

        starts = []
        ends = []

        delt = snrs[iref].delta_t
        tmin = tcmin + dt - delt
        tmax = tcmax + dt + delt
        if hasattr(self, "tstart"):
            tmin = self.tstart[iref]
            tmax = self.tend[iref]

        # Make sure we draw from times within prior and that have enough
        # SNR calculated to do later interpolation
        starts.append(max(tmin, snrs[iref].start_time + delt))
        ends.append(min(tmax, snrs[iref].end_time - delt * 2))

        idels = {}
        for ifo in ifos[1:]:
            dti = d[ifo].time_delay_from_detector(dref, ra, dec, tcave)
            idel = round(dti / snrs[iref].delta_t) * snrs[iref].delta_t
            idels[ifo] = idel

            starts.append(snrs[ifo].start_time - idel)
            ends.append(snrs[ifo].end_time - idel)
        start = max(starts)
        end = min(ends)
        if end <= start:
            return

        # get the weights
        snr = snrs[iref].time_slice(start, end, mode="nearest")
        logweight = _squared_norm_values(snr)
        for ifo in ifos[1:]:
            idel = idels[ifo]
            snrv = snrs[ifo].time_slice(
                snr.start_time + idel, snr.end_time + idel, mode="nearest"
            )
            logweight += _squared_norm_values(snrv)
        logweight /= 2.0
        logweight = _normalize_logweights(logweight)

        # Draw proportional to the incoherent likelihood
        # Draw first which time sample
        tci = draw_sample(logweight, size=vsamples, host=False)
        # Second draw a subsample size offset so that all times are covered
        tct = numpy.random.uniform(-snr.delta_t / 2.0, snr.delta_t / 2.0, size=vsamples)
        # Public GPS times need host float64 precision, notably on MPS.
        time_indices = _host_indices(tci)
        tc = tct + time_indices * snr.delta_t + float(snr.start_time) - dt

        # Update the current proposed times and the marginalization values
        # assumes uniform prior!
        logw = -_selected_values(logweight, tci, host=False) + numpy.log(
            1.0 / len(logweight)
        )
        self.marginalize_vector_params["tc"] = tc
        self.marginalize_vector_params["logw_partial"] = logw

        if self._current_params is not None:
            # Update the importance weights for each vector sample
            self._current_params.update(self.marginalize_vector_params)
            self.marginalize_vector_weights = _add_values(
                self.marginalize_vector_weights, logw
            )

        return self.marginalize_vector_params

    def draw_sky_times(self, snrs, size=None):
        """Draw ra, dec, and tc together using SNR timeseries to determine
        monte-carlo weights.
        """
        # First setup
        # precalculate dense sky grid and make dict and or array of the results
        ifos = list(snrs.keys())
        if hasattr(self, "keep_ifos"):
            ifos = self.keep_ifos
        ikey = "".join(ifos)

        vsamples = size if size is not None else self.vsamples

        # No good SNR peaks, go with prior draw
        if len(ifos) == 0:
            self.marginalize_vector_params["logw_partial"] = numpy.zeros(vsamples)
            return

        def make_init():
            self.snr_params = ["tc", "ra", "dec"]
            size = self.marginalize_sky_initial_samples
            logging.info("drawing samples: %s", size)
            ra = self.marginalized_vector_priors["ra"].rvs(size=size)["ra"]
            dec = self.marginalized_vector_priors["dec"].rvs(size=size)["dec"]
            tcmin, tcmax = self.marginalized_vector_priors["tc"].bounds["tc"]
            tcave = (tcmax + tcmin) / 2.0
            d = {ifo: Detector(ifo, reference_time=tcave) for ifo in self.data}

            # What data structure to hold times? Dict of offset -> list?
            logging.info("sorting into time delay dict")
            dts = []
            for i in range(len(ifos) - 1):
                dt = d[ifos[0]].time_delay_from_detector(d[ifos[i + 1]], ra, dec, tcave)
                dt = numpy.rint(dt / snrs[ifos[0]].delta_t)
                dts.append(dt)

            fp, fc, dtc = {}, {}, {}
            for ifo in self.data:
                fp[ifo], fc[ifo] = d[ifo].antenna_pattern(ra, dec, 0.0, tcave)
                dtc[ifo] = d[ifo].time_delay_from_earth_center(ra, dec, tcave)

            dmap = {}
            for i, t in enumerate(tqdm.tqdm(zip(*dts))):
                if t not in dmap:
                    dmap[t] = []
                dmap[t].append(i)

            if len(ifos) == 1:
                dmap[()] = numpy.arange(0, size, 1).astype(int)

            # Sky prior by bin
            bin_prior = {t: len(dmap[t]) / size for t in dmap}

            return dmap, tcmin, tcmax, fp, fc, ra, dec, dtc, bin_prior

        if not hasattr(self, "tinfo"):
            self.tinfo = {}

        if ikey not in self.tinfo:
            logging.info("pregenerating sky pointings")
            self.tinfo[ikey] = make_init()

        dmap, tcmin, tcmax, fp, fc, ra, dec, dtc, bin_prior = self.tinfo[ikey]

        # draw times from each snr time series
        # Is it worth doing this if some detector has low SNR?
        sref = None
        draw_weights = []
        sample_offsets = []
        for ifo in ifos:
            snr = snrs[ifo]
            tmin, tmax = tcmin - EARTH_RADIUS, tcmax + EARTH_RADIUS
            if hasattr(self, "tstart"):
                tmin = self.tstart[ifo]
                tmax = self.tend[ifo]

            start = max(tmin, snr.start_time + snr.delta_t)
            end = min(tmax, snr.end_time - snr.delta_t * 2)
            snr = snr.time_slice(start, end, mode="nearest")

            w = _squared_norm_values(snr) / 2.0
            if sref is not None:
                delt = float(snr.start_time - sref.start_time)
                sample_offsets.append(round(delt / sref.delta_t))
            else:
                sref = snr
                sample_offsets.append(0)
            draw_weights.append(w)

        iref, dx, selected_weights = _draw_sky_time_indices(
            draw_weights, sample_offsets, vsamples
        )
        mcweight = None
        for selected_weight in selected_weights:
            mcweight = _add_values(mcweight, selected_weight)

        mcweight = _normalize_logweights(mcweight)

        # check if delay is in dict, if not, throw out
        ti = []
        ix = []
        wi = []
        rand = numpy.random.uniform(0, 1, size=vsamples)
        for i in range(vsamples):
            t = tuple(x[i] for x in dx)
            if t in dmap:
                randi = int(rand[i] * (len(dmap[t])))
                ix.append(dmap[t][randi])
                wi.append(bin_prior[t])
                ti.append(i)

        # If we had really poor efficiency at finding a point, we should
        # give up and just use the original random draws
        if len(ix) < 0.05 * vsamples:
            self.marginalize_vector_params["logw_partial"] = numpy.zeros(vsamples)
            return

        # fill back to fixed size with repeat samples
        # sample order is random, so this should be OK statistically
        ix = numpy.resize(numpy.array(ix, dtype=int), vsamples)
        self.sample_idx = ix
        self.precalc_antenna_factors = fp, fc, dtc
        resize_factor = len(ti) / vsamples

        ra = ra[ix]
        dec = dec[ix]
        dtc = {ifo: dtc[ifo][ix] for ifo in dtc}

        ti = numpy.resize(numpy.array(ti, dtype=int), vsamples)
        wi = numpy.resize(numpy.array(wi), vsamples)

        # Second draw a subsample size offset so that all times are covered
        tct = numpy.random.uniform(-snr.delta_t / 2.0, snr.delta_t / 2.0, size=len(ti))

        tc = tct + iref[ti] * snr.delta_t + float(sref.start_time) - dtc[ifos[0]]

        # Update the current proposed times and the marginalization values
        # There's an overall normalization here which may introduce a constant
        # factor at the moment.
        selected_mcweight = _selected_values(mcweight, ti, host=False)
        logw_sky = _add_values(
            -selected_mcweight,
            numpy.log(wi) - numpy.log(resize_factor),
        )

        self.marginalize_vector_params["tc"] = tc
        self.marginalize_vector_params["ra"] = ra
        self.marginalize_vector_params["dec"] = dec
        self.marginalize_vector_params["logw_partial"] = logw_sky

        if self._current_params is not None:
            # Update the importance weights for each vector sample
            self._current_params.update(self.marginalize_vector_params)
            self.marginalize_vector_weights = _add_values(
                self.marginalize_vector_weights, logw_sky
            )

        return self.marginalize_vector_params

    def setup_peak_lock(
        self,
        sample_rate=4096,
        snrs=None,
        peak_lock_snr=None,
        peak_lock_ratio=1e4,
        peak_lock_region=4,
        **kwargs,
    ):
        """Determine where to constrain marginalization based on
        the observed reference SNR peaks.

        Parameters
        ----------
        sample_rate : float
            The SNR sample rate
        snrs : Dict of SNR time series
            Either provide this or the model needs a function
            to get the reference SNRs.
        peak_lock_snr: float
            The minimum SNR to bother restricting from the prior range
        peak_lock_ratio: float
            The likelihood ratio (not log) relative to the peak to
            act as a threshold bounding region.
        peak_lock_region: int
            Number of samples to inclue beyond the strict region
            determined by the relative likelihood
        """

        if "tc" not in self.marginalized_vector_priors:
            return

        tcmin, tcmax = self.marginalized_vector_priors["tc"].bounds["tc"]
        tstart = tcmin - EARTH_RADIUS
        tmax = tcmax - tcmin + EARTH_RADIUS * 2.0
        num_samples = int(tmax * sample_rate)
        self.tstart = {ifo: tstart for ifo in self.data}
        self.num_samples = {ifo: num_samples for ifo in self.data}

        if snrs is None:
            if not hasattr(self, "ref_snr"):
                raise ValueError("Model didn't have a reference SNR!")
            snrs = self.ref_snr

        # Restrict the time range for constructing SNR time series
        # to identifiable peaks
        if peak_lock_snr is not None:
            peak_lock_snr = float(peak_lock_snr)
            peak_lock_ratio = float(peak_lock_ratio)
            peak_lock_region = int(peak_lock_region)

            for ifo in snrs:
                s = max(tstart, snrs[ifo].start_time)
                e = min(tstart + tmax, snrs[ifo].end_time)
                z = snrs[ifo].time_slice(s, e, mode="nearest")
                peak_snr, imax = z.abs_max_loc()
                start_time = float(z.start_time)
                peak_time = start_time + imax * z.delta_t

                logging.info(
                    "%s: Max Ref SNR Peak of %s at %s", ifo, peak_snr, peak_time
                )

                if peak_snr > peak_lock_snr:
                    target = peak_snr**2.0 / 2.0 - numpy.log(peak_lock_ratio)
                    target = (target * 2.0) ** 0.5

                    first, last = _threshold_extent(z, target)
                    ts = start_time + first * z.delta_t - peak_lock_region / sample_rate
                    te = start_time + last * z.delta_t + peak_lock_region / sample_rate
                    self.tstart[ifo] = ts
                    self.num_samples[ifo] = int((te - ts) * sample_rate)

            # Check times are commensurate with each other
            for ifo in snrs:
                ts = self.tstart[ifo]
                te = ts + self.num_samples[ifo] / sample_rate

                for ifo2 in snrs:
                    if ifo == ifo2:
                        continue
                    ts2 = self.tstart[ifo2]
                    te2 = ts2 + self.num_samples[ifo2] / sample_rate
                    det = Detector(ifo)
                    dt = Detector(ifo2).light_travel_time_to_detector(det)

                    ts = max(ts, ts2 - dt)
                    te = min(te, te2 + dt)

                self.tstart[ifo] = ts
                self.num_samples[ifo] = int((te - ts) * sample_rate) + 1
                logging.info(
                    "%s: use region %s-%s, %s points",
                    ifo,
                    ts,
                    te,
                    self.num_samples[ifo],
                )

        self.tend = self.tstart.copy()
        for ifo in snrs:
            self.tend[ifo] += self.num_samples[ifo] / sample_rate

    def reconstruct(self, rec=None, seed=None, set_loglr=None):
        """Reconstruct the distance or vectored marginalized parameter
        of this class.
        """
        if seed is not None:
            numpy.random.seed(seed)

        if rec is None:
            rec = {}

        if set_loglr is None:

            def get_loglr():
                p = self.current_params.copy()
                p.update(rec)
                self.update(**p)
                return self.loglr

        else:
            get_loglr = set_loglr

        if self.marginalize_vector_params:
            logging.debug("Reconstruct vector")
            self.reconstruct_vector = True
            self.reset_vector_params()
            loglr = get_loglr()
            xl = draw_sample(loglr + self.marginalize_vector_weights)
            for k in self.marginalize_vector_params:
                rec[k] = self.marginalize_vector_params[k][xl]
            self.reconstruct_vector = False

        if self.distance_marginalization:
            logging.debug("Reconstruct distance")
            # call likelihood to get vector output
            self.reconstruct_distance = True
            _, weights = self.distance_marginalization
            loglr = get_loglr()
            xl = draw_sample(_weighted_loglr(loglr, weights))
            rec["distance"] = self.dist_locs[xl]
            self.reconstruct_distance = False

        if self.marginalize_phase:
            logging.debug("Reconstruct phase")
            self.reconstruct_phase = True
            s, h = get_loglr()
            # This assumes that the template was conjugated in inner products
            phasev, loglr = _phase_reconstruction_values(s, h)
            xl = draw_sample(loglr)
            rec["coa_phase"] = _selected_scalar(phasev, xl)
            self.reconstruct_phase = False

        rec["loglr"] = _selected_scalar(loglr, xl)
        rec["loglikelihood"] = self.lognl + rec["loglr"]
        return rec


def _distance_interpolant(native, spline):
    from .tools_jax import rect_bivariate_spline_evaluator

    evaluate = rect_bivariate_spline_evaluator(spline)

    def wrapper(x, y, bounds_check=True):
        if _jax_array(x) is not None or _jax_array(y) is not None:
            return evaluate(x, y, bounds_check=bounds_check)
        return native(x, y, bounds_check=bounds_check)

    wrapper._jax_evaluate = evaluate
    return wrapper
