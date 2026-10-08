"""Scalar boundaries for JAX inference models."""

import numpy
from pycbc.types.backend import backend_array
from .base import BaseModel, ModelStats
from .base_data import BaseDataModel


def _jax_array(value):
    """Return the array backing a JAX/PyCBC value, if present."""
    return backend_array(value, "jax")


def _replace_nan_with_neginf(value):
    """Replace a NaN model statistic without moving JAX data to the host."""
    arr = _jax_array(value)
    if arr is None:
        return -numpy.inf if numpy.isnan(value) else value

    import jax.numpy as jnp

    return jnp.where(jnp.isnan(arr), jnp.array(-numpy.inf, dtype=arr.dtype), arr)


def _is_neginf_scalar(value):
    """Return whether a scalar model statistic is negative infinity."""
    arr = _jax_array(value)
    if arr is None:
        return value == -numpy.inf
    if arr.size != 1:
        raise ValueError("model statistics must be scalar values")

    import jax.numpy as jnp

    return bool(jnp.isneginf(arr))


def _public_stat_value(value):
    """Materialize a scalar JAX statistic at the public stats boundary."""
    arr = _jax_array(value)
    if arr is value and arr.ndim == 0:
        return arr.item()
    return value


class JAXModelStats(ModelStats):
    """JAX implementation of :class:`ModelStats`."""

    def getstats(self, names, default=numpy.nan):
        """Get the requested stats as a tuple.

        If a requested stat is not an attribute (implying it hasn't been
        stored), then the default value is returned for that stat.
        Device-resident scalar JAX values are materialized here so sampler
        and serialization callers retain the established scalar interface.

        Parameters
        ----------
        names : list of str
            The names of the stats to get.
        default : float, optional
            What to return if a requested stat is not an attribute of self.
            Default is ``numpy.nan``.

        Returns
        -------
        tuple
            A tuple of the requested stats.
        """
        return tuple(_public_stat_value(getattr(self, n, default)) for n in names)

    def getstatsdict(self, names, default=numpy.nan):
        """Get the requested stats as a dictionary.

        If a requested stat is not an attribute (implying it hasn't been
        stored), then the default value is returned for that stat.

        Parameters
        ----------
        names : list of str
            The names of the stats to get.
        default : float, optional
            What to return if a requested stat is not an attribute of self.
            Default is ``numpy.nan``.

        Returns
        -------
        dict
            A dictionary of the requested stats.
        """
        return {n: _public_stat_value(getattr(self, n, default)) for n in names}


class JAXBaseModel(BaseModel):
    """JAX implementation of :class:`BaseModel`."""

    def _logprior(self):
        """Calculates the log prior at the current parameters."""
        logj = self.logjacobian
        logp = self.prior_distribution(**self.current_params) + logj
        return _replace_nan_with_neginf(logp)

    @property
    def logposterior(self):
        """Returns the log of the posterior of the current parameter values.

        The logprior is calculated first. If the logprior returns ``-inf``
        (possibly indicating a non-physical point), then the ``loglikelihood``
        is not called.
        """
        logp = self.logprior
        if _is_neginf_scalar(logp):
            return logp
        else:
            return logp + self.loglikelihood


class JAXBaseDataModel(BaseDataModel, JAXBaseModel):
    """JAX implementation of :class:`BaseDataModel`."""

    @property
    def logplr(self):
        """Returns the log of the prior-weighted likelihood ratio at the
        current parameter values.

        The logprior is calculated first. If the logprior returns ``-inf``
        (possibly indicating a non-physical point), then ``loglr`` is not
        called.
        """
        logp = self.logprior
        if _is_neginf_scalar(logp):
            return logp
        return logp + self.loglr
