"""JAX detector-frame waveform generator implementations."""

import os
import numpy
from pycbc import strain
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.types.backend import backend_array, wrap_backend_array
from pycbc.waveform.utils import apply_fd_time_shift, ceilpow2
from . import ringdown, supernovae, waveform
from .waveform import FailedWaveformError

from .generator import (
    BaseGenerator,
    BaseCBCGenerator,
    FDomainCBCGenerator,
    FDomainCBCModesGenerator,
    TDomainCBCGenerator,
    TDomainCBCModesGenerator,
    FDomainDetFrameGenerator,
    FDomainMassSpinRingdownGenerator,
    FDomainFreqTauRingdownGenerator,
    TDomainMassSpinRingdownGenerator,
    TDomainFreqTauRingdownGenerator,
    TDomainSupernovaeGenerator,
)

from . import generator as original_generator

_LOW_PRECISION_ABSOLUTE_TIME = 2**24


def _jax_value(value):
    """Return JAX storage through the public backend protocol."""
    return backend_array(value, "jax")


def _validate_absolute_time_precision(reference_time, epoch):
    """Reject absolute times whose fractional part is already unrecoverable."""
    jax_arr = _jax_value(reference_time)
    if jax_arr is not None:
        import jax.numpy as jnp

        large_absolute_time = abs(float(epoch)) >= (_LOW_PRECISION_ABSOLUTE_TIME)
        if not large_absolute_time and jax_arr.size:
            large_absolute_time = bool(
                jnp.any(jnp.abs(jax_arr) >= _LOW_PRECISION_ABSOLUTE_TIME)
            )
        if large_absolute_time and jax_arr.dtype != jnp.float64:
            raise ValueError(
                "Absolute GPS time tensors must use float64; lower "
                "precision dtypes cannot represent detector arrival times"
            )
        return large_absolute_time

    return abs(float(epoch)) >= (_LOW_PRECISION_ABSOLUTE_TIME)


def _arrival_time_and_shift(reference_time, offset, epoch, extra_shift=0.0):
    """Return a precise arrival time and its shift relative to ``epoch``.

    The large absolute time is kept in float64 while the phase shift is
    centered before it is cast to a waveform's lower-precision device dtype.
    """
    from pycbc.types.array_jax import _reference_enabled, to_jax

    if _reference_enabled("time_shift"):
        import jax

        anchor = _jax_value(reference_time)
        if anchor is None:
            anchor = _jax_value(offset)
        if isinstance(anchor, jax.core.Tracer):
            raise RuntimeError("Native time-shift validation cannot run inside jax.jit")
        arrival = numpy.asarray(reference_time) + numpy.asarray(offset)
        relative = (arrival + extra_shift) - float(epoch)
        if anchor is not None:
            return to_jax(arrival, device=anchor.device), to_jax(
                relative, device=anchor.device
            )
        return arrival, relative
    large_absolute_time = _validate_absolute_time_precision(reference_time, epoch)

    reference_jax = _jax_value(reference_time)
    offset_jax = _jax_value(offset)
    if reference_jax is not None or offset_jax is not None:
        import jax.numpy as jnp

        anchor = reference_jax if reference_jax is not None else offset_jax
        work_dtype = jnp.float64 if large_absolute_time else anchor.dtype
        if not jnp.issubdtype(anchor.dtype, jnp.floating):
            work_dtype = jnp.float64

        if reference_jax is None:
            reference_on_device = jnp.asarray(reference_time, dtype=work_dtype)
            centered_reference = jnp.asarray(
                numpy.asarray(reference_time, dtype=numpy.float64) - float(epoch),
                dtype=work_dtype,
            )
        else:
            reference_on_device = jnp.asarray(reference_jax, dtype=work_dtype)
            centered_reference = jnp.asarray(
                reference_jax - float(epoch),
                dtype=work_dtype,
            )

        offset_on_device = jnp.asarray(
            offset_jax if offset_jax is not None else offset,
            dtype=work_dtype,
        )
        arrival_time = reference_on_device + offset_on_device
        relative_shift = centered_reference + offset_on_device + extra_shift
        return arrival_time, relative_shift

    arrival_time = reference_time + offset
    relative_shift = (reference_time - epoch) + offset + extra_shift
    return arrival_time, relative_shift


def _detector_time_offset(detector, reference_time, ra, dec, reference_frame):
    """Return a detector arrival offset without adding it to absolute GPS."""
    from pycbc.detector import Detector

    try:
        instance_override = "arrival_time" in vars(detector)
    except TypeError:
        instance_override = False
    class_arrival_time = getattr(type(detector), "arrival_time", None)
    if instance_override or (
        class_arrival_time is not None
        and class_arrival_time is not Detector.arrival_time
    ):
        return (
            detector.arrival_time(reference_time, ra, dec, reference_frame)
            - reference_time
        )

    if reference_frame == "geocentric" and hasattr(
        detector, "time_delay_from_earth_center"
    ):
        return detector.time_delay_from_earth_center(ra, dec, reference_time)
    if reference_frame == getattr(detector, "name", None):
        return 0.0
    if hasattr(detector, "time_delay_from_detector"):
        return detector.time_delay_from_detector(
            Detector(reference_frame), ra, dec, reference_time
        )

    # Lightweight detector doubles historically only implemented
    # ``arrival_time``. Preserve that compatibility for scalar tests/users.
    return (
        detector.arrival_time(reference_time, ra, dec, reference_frame) - reference_time
    )


def _has_sample_axis(value):
    """Return whether a parameter value has a non-scalar shape."""
    jax_arr = _jax_value(value)
    if jax_arr is not None:
        return jax_arr.ndim != 0
    return bool(numpy.shape(value))


class JAXBaseGenerator(BaseGenerator):
    _gdecorator = BaseGenerator._gdecorator
    """JAX implementation of :class:`BaseGenerator`."""

    @_gdecorator
    def _generate_from_current(self):
        """Generates a waveform from the current parameters."""
        try:
            new_waveform = self.generator(
                **(
                    _radiation_parameters(self.current_params)
                    if isinstance(self, JAXBaseCBCGenerator)
                    else self.current_params
                )
            )
            return new_waveform
        except RuntimeError as e:
            if self.record_failures:
                from pycbc.io.hdf import HFile, dump_state

                if self.mpi_enabled:
                    outname = "failed/params_%s.hdf" % self.mpi_rank
                else:
                    outname = "failed/params.hdf"

                if not os.path.exists("failed"):
                    os.makedirs("failed")

                with HFile(outname) as f:
                    dump_state(
                        self.current_params,
                        f,
                        dsetname=str(original_generator.failed_counter),
                    )
                    original_generator.failed_counter += 1

            # we'll get a RuntimeError if lalsimulation failed to generate
            # the waveform for whatever reason
            strparams = " | ".join(
                ["{}: {}".format(p, str(val)) for p, val in self.current_params.items()]
            )
            raise FailedWaveformError(
                "Failed to generate waveform with "
                "parameters:\n{}\nError was: {}".format(strparams, e)
            ) from e


class JAXBaseCBCGenerator(BaseCBCGenerator, JAXBaseGenerator):
    """JAX implementation of :class:`BaseCBCGenerator`."""

    pass


class JAXFDomainCBCGenerator(FDomainCBCGenerator, JAXBaseCBCGenerator):
    """JAX implementation of :class:`FDomainCBCGenerator`."""

    pass


class JAXFDomainCBCModesGenerator(FDomainCBCModesGenerator, JAXBaseCBCGenerator):
    """JAX implementation of :class:`FDomainCBCModesGenerator`."""

    pass


class JAXTDomainCBCGenerator(TDomainCBCGenerator, JAXBaseCBCGenerator):
    """JAX implementation of :class:`TDomainCBCGenerator`."""


class JAXTDomainCBCModesGenerator(TDomainCBCModesGenerator, JAXBaseCBCGenerator):
    """JAX implementation of :class:`TDomainCBCModesGenerator`."""


class JAXFDomainDetFrameGenerator(FDomainDetFrameGenerator):
    """JAX implementation of :class:`FDomainDetFrameGenerator`."""

    def generate(self, **kwargs):
        """Generates a waveform, applies a time shift and the detector response
        function from the given kwargs.
        """
        next_params = self.current_params.copy()
        next_params.update(kwargs)
        batched_location_args = sorted(
            name
            for name in self.location_args | {"tc_ref_frame"}
            if name in next_params and _has_sample_axis(next_params[name])
        )
        if batched_location_args:
            raise ValueError(
                "FDomainDetFrameGenerator.generate does not return a batch "
                "container; detector-frame parameters must be scalar. Got: "
                + ", ".join(batched_location_args)
            )
        if "tc" in next_params:
            _validate_absolute_time_precision(next_params["tc"], self._epoch)
        self.current_params.update(kwargs)
        rfparams = {
            param: self.current_params[param]
            for param in kwargs
            if param not in self.location_args
        }
        hp, hc = self.rframe_generator.generate(**rfparams)
        from pycbc.types.array import _convert_to_scheme

        _convert_to_scheme(hp)
        _convert_to_scheme(hc)
        if len(hp.shape) != 1 or len(hc.shape) != 1:
            raise ValueError(
                "FDomainDetFrameGenerator.generate does not return a batch "
                "container; radiation-frame waveforms must be one-dimensional"
            )
        if isinstance(hp, TimeSeries):
            df = self.current_params["delta_f"]
            hp = hp.to_frequencyseries(delta_f=df)
            hc = hc.to_frequencyseries(delta_f=df)
            # time-domain waveforms will not be shifted so that the peak amp
            # happens at the end of the time series (as they are for f-domain),
            # so we add an additional shift to account for it
            tshift = 1.0 / df - abs(hp._epoch)
        else:
            tshift = 0.0
        hp._epoch = hc._epoch = self._epoch
        h = {}
        if self.detector_names != ["RF"]:
            ra = self.current_params["ra"]
            dec = self.current_params["dec"]
            ref_tc = self.current_params["tc"]
            pol = self.current_params["polarization"]
            refframe = self.current_params.get("tc_ref_frame", "geocentric")

            hp_jax = backend_array(hp, "jax")
            hc_jax = backend_array(hc, "jax")
            use_jax_fused = hp_jax is not None and hc_jax is not None

            if use_jax_fused:
                from .utils_jax import fused_detector_strain_fd_jax

                fp_list = []
                fc_list = []
                dt_list = []
                det_list = []
                for detname, det in self.detectors.items():
                    offset = _detector_time_offset(det, ref_tc, ra, dec, refframe)
                    tc, dt = _arrival_time_and_shift(
                        ref_tc, offset, self._epoch, tshift
                    )
                    fp, fc = det.antenna_pattern(ra, dec, pol, tc)
                    fp_list.append(fp)
                    fc_list.append(fc)
                    dt_list.append(dt)
                    det_list.append(detname)

                strains = fused_detector_strain_fd_jax(
                    hp_jax, hc_jax, fp_list, fc_list, dt_list, hp.delta_f
                )
                for i, detname in enumerate(det_list):
                    series = FrequencySeries(
                        wrap_backend_array(strains[i]),
                        delta_f=hp.delta_f,
                        epoch=self._epoch,
                        copy=False,
                    )
                    if self.recalib:
                        series = self.recalib[detname].map_to_adjust(
                            series, **self.current_params
                        )
                    h[detname] = series
            else:
                for detname, det in self.detectors.items():
                    tc = det.arrival_time(ref_tc, ra, dec, refframe)
                    # Evaluate the detector tensor at the arrival time.  The
                    # sidereal response changes between the reference and
                    # detector-frame times, even though the difference is
                    # normally only milliseconds.
                    fp, fc = det.antenna_pattern(ra, dec, pol, tc)
                    thish = fp * hp + fc * hc
                    h[detname] = apply_fd_time_shift(thish, tc + tshift, copy=False)
                    if self.recalib:
                        # recalibrate with given calibration model
                        h[detname] = self.recalib[detname].map_to_adjust(
                            h[detname], **self.current_params
                        )
        else:
            # no detector response, just use the + polarization
            if "tc" in self.current_params:
                hp = apply_fd_time_shift(
                    hp, self.current_params["tc"] + tshift, copy=False
                )
            h["RF"] = hp
        if self.gates is not None:
            # resize all to nearest power of 2
            for d in h.values():
                d.resize(ceilpow2(len(d) - 1) + 1)
            h = strain.apply_gates_to_fd(h, self.gates)
        return h


def get_td_generator(approximant, modes=False):
    """Returns the time-domain generator for the given approximant."""
    if approximant in waveform.td_approximants():
        if modes:
            return JAXTDomainCBCModesGenerator
        return JAXTDomainCBCGenerator

    from pycbc.types.array_jax import _reference_enabled

    if not _reference_enabled("waveform"):
        raise ValueError(
            "JAX waveform generation requires a diffGW-supported approximant"
        )
    if approximant in ringdown.ringdown_td_approximants:
        if approximant == "TdQNMfromFinalMassSpin":
            return TDomainMassSpinRingdownGenerator
        return TDomainFreqTauRingdownGenerator

    if approximant in supernovae.supernovae_td_approximants:
        return TDomainSupernovaeGenerator

    raise ValueError(f"No time-domain generator found for approximant: {approximant}")


def get_fd_generator(approximant, modes=False):
    """Returns the frequency-domain generator for the given approximant."""
    if approximant in waveform.fd_approximants():
        if modes:
            return JAXFDomainCBCModesGenerator
        return JAXFDomainCBCGenerator

    from pycbc.types.array_jax import _reference_enabled

    if not _reference_enabled("waveform"):
        raise ValueError(
            "JAX waveform generation requires a diffGW-supported approximant"
        )
    if approximant in ringdown.ringdown_fd_approximants:
        if approximant == "FdQNMfromFinalMassSpin":
            return FDomainMassSpinRingdownGenerator
        return FDomainFreqTauRingdownGenerator

    raise ValueError(
        f"No frequency-domain generator found for approximant: {approximant}"
    )


def implementation(cls):
    mapping = {FDomainDetFrameGenerator: JAXFDomainDetFrameGenerator}
    return mapping.get(cls)


def _radiation_parameters(params):
    """Remove detector-frame parameters from a CBC radiation-frame request."""
    return {
        name: value
        for name, value in params.items()
        if name not in {"tc", "ra", "dec", "polarization", "tc_ref_frame"}
    }
