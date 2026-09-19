# Copyright (C) 2020 Alexander Nitz
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
"""
This modules provides classes for evaluating distributions based on a fixed
set of points
"""
import logging
import operator
import numpy
import numpy.random

from pycbc import VARARGS_DELIM
from pycbc.distributions import bounded

logger = logging.getLogger('pycbc.distributions.fixedsamples')


class FixedSamples(object):
    """
    A distribution consisting of a collection of a large number of fixed points.
    Only these values can be drawn from, so the number of points may need to be
    large to properly reflect the paramter space. This distribution is intended
    to aid in using nested samplers for semi-abitrary or complicated
    distributions where it is possible to provide or draw samples but less
    straightforward to provide an analytic invcdf. This class numerically
    approximates the invcdf for 1 or 2 dimensional distributions
    (but no higher).

    Parameters
    ----------
    params :
        This of parameters this distribution should use
    samples : dict of arrays or FieldArray
        Sampled points of the distribution. May contain transformed parameters
        which are different from the original distribution. If so, an inverse
        mapping is provided to associate points with other parameters provided.
    """

    name = "fixed_samples"

    def __init__(self, params, samples):
        self.params = params
        self.samples = samples
        self._jax_reference = None
        self._jax_states = {}

        jax, reference = bounded._jax_module_and_reference(
            self.samples[p] for p in self.params
        )
        if jax is not None:
            self._jax_reference = reference
            state = self._jax_state(reference)
            self.p1 = state["p1"]
            self.sort = state["sort"]
            self.p1sorted = state["p1sorted"]
        else:
            self.p1 = self.samples[params[0]]
            self.sort = self.p1.argsort()
            self.p1sorted = self.p1[self.sort]
            assert len(numpy.unique(self.p1)) == len(self.p1)

        self.frac = len(self.p1) ** 0.5 / len(self.p1)

        if len(params) > 2:
            raise ValueError(
                "Only one or two parameters supported for fixed sample distribution"
            )

    def _jax_state(self, reference):
        """Return fixed samples and their ordering on JAX backend."""
        import jax.numpy as jnp
        dtype = reference.dtype
        if not (jnp.issubdtype(dtype, jnp.floating) or jnp.issubdtype(dtype, jnp.complexfloating)):
            dtype = jnp.float64
        try:
            return self._jax_states[dtype]
        except KeyError:
            pass

        samples = {
            p: jnp.asarray(self.samples[p], dtype=dtype)
            for p in self.params
        }
        p1 = samples[self.params[0]]
        if any(len(samples[p]) != len(p1) for p in self.params):
            raise ValueError("fixed-sample parameter arrays must have equal length")
        if len(jnp.unique(p1)) != len(p1):
            raise AssertionError("first fixed-sample parameter must be unique")
        sort = jnp.argsort(p1)
        state = {
            "samples": samples,
            "p1": p1,
            "sort": sort,
            "p1sorted": p1[sort],
        }
        self._jax_states[dtype] = state
        return state

    @staticmethod
    def _draw_shape(size):
        if size is None:
            return ()
        try:
            return tuple(operator.index(value) for value in size)
        except TypeError:
            return (operator.index(size),)

    def rvs(self, size=1, **kwds):
        "Draw random value"
        i = numpy.random.randint(0, high=len(self.p1), size=size)
        return {p: self.samples[p][i] for p in self.params}

    def cdfinv(self, **original):
        """Map unit cube to parameters in the space"""
        jax, reference = bounded._jax_module_and_reference(original.values())
        if jax is None and self._jax_reference is not None:
            reference = self._jax_reference
            jax, _ = bounded._jax_module_and_reference((reference,))
        if jax is not None:
            return self._cdfinv_jax(original, reference)

        new = {}

        #First dimension
        u1 = original[self.params[0]]
        i1 = int(round(u1 * len(self.p1)))
        if i1 >= len(self.p1):
            i1 = len(self.p1) - 1
        if i1 < 0:
            i1 = 0
        new[self.params[0]] = p1v = self.p1sorted[i1]
        if len(self.params) == 1:
            return new

        # possible second dimension, probably shouldn't
        # do more dimensions than this
        u2 = original[self.params[1]]
        l = numpy.searchsorted(self.p1sorted, p1v * (1 - self.frac))
        r = numpy.searchsorted(self.p1sorted, p1v * (1 + self.frac))
        if r < l:
            l, r = r, l

        region = numpy.array(self.sort[l:r], ndmin=1)
        p2 = self.samples[self.params[1]]
        p2part = numpy.array(p2[region], ndmin=1)
        l = p2part.argsort()
        p2part = numpy.array(p2part[l], ndmin=1)

        i2 = int(round(u2 * len(p2part)))
        if i2 >= len(p2part):
            i2 = len(p2part) - 1
        if i2 < 0:
            i2 = 0
        new[self.params[1]] = p2part[i2]

        p1part = numpy.array(self.p1[region[l]], ndmin=1)
        new[self.params[0]] = p1part[i2]
        return new

    def _cdfinv_jax(self, original, reference):
        """Map unit-cube arrays without staging fixed samples on the host."""
        import jax.numpy as jnp
        state = self._jax_state(reference)
        p1 = state["p1"]
        p1sorted = state["p1sorted"]
        sort = state["sort"]
        u1 = jnp.asarray(original[self.params[0]], dtype=p1.dtype)
        i1 = jnp.round(u1 * len(p1)).astype(int)
        i1 = jnp.clip(i1, 0, len(p1) - 1)
        if len(self.params) == 1:
            return {self.params[0]: p1sorted[i1]}

        p2 = state["samples"][self.params[1]]
        u2 = jnp.asarray(original[self.params[1]], dtype=p1.dtype)
        u1, u2 = jnp.broadcast_arrays(u1, u2)
        shape = u1.shape
        i1 = jnp.round(u1.reshape(-1) * len(p1)).astype(int)
        i1 = jnp.clip(i1, 0, len(p1) - 1)
        u2 = u2.reshape(-1)
        positions = jnp.arange(len(p1))
        selected = []
        for index in range(i1.size):
            p1v = p1sorted[i1[index]]
            left = jnp.searchsorted(p1sorted, p1v * (1 - self.frac))
            right = jnp.searchsorted(p1sorted, p1v * (1 + self.frac))
            low = jnp.minimum(left, right)
            high = jnp.maximum(left, right)
            region = sort[(positions >= low) & (positions < high)]
            if region.size == 0:
                raise IndexError("fixed-sample inverse CDF selected no points")
            order = jnp.argsort(p2[region])
            i2 = jnp.round(u2[index] * len(region)).astype(int)
            i2 = jnp.clip(i2, 0, len(region) - 1)
            selected.append(region[order[i2]])

        if selected:
            selected = jnp.stack(selected).reshape(shape)
        else:
            selected = jnp.empty(shape, dtype=int)
        return {
            self.params[0]: p1[selected],
            self.params[1]: p2[selected],
        }

    def apply_boundary_conditions(self, **params):
        """ Apply boundary conditions (none here) """
        return params

    def __call__(self, **kwds):
        """ Dummy function, not the actual pdf """
        return 0

    @classmethod
    def from_config(cls, cp, section, tag):
        """ Return instance based on config file

        Return a new instance based on the config file. This will draw from
        a single distribution section provided in the config file and
        apply a single transformation section if desired. If a transformation
        is applied, an inverse mapping is also provided for use in the config
        file.
        """
        from pycbc.distributions import read_distributions_from_config
        from pycbc.transforms import (read_transforms_from_config,
                                      apply_transforms, BaseTransform)
        from pycbc.transforms import transforms as global_transforms

        params = tag.split(VARARGS_DELIM)
        subname = cp.get_opt_tag(section, 'subname', tag)
        size = cp.get_opt_tag(section, 'sample-size', tag)

        distsec = '{}_sample'.format(subname)
        dist = read_distributions_from_config(cp, section=distsec)
        if len(dist) > 1:
            raise ValueError("Fixed sample distrubtion only supports a single"
                             " distribution to sample from.")

        logger.info('Drawing samples for fixed sample distribution:%s', params)
        samples = dist[0].rvs(size=int(float(size)))
        samples = {p: samples[p] for p in samples.dtype.names}

        transec = '{}_transform'.format(subname)
        trans = read_transforms_from_config(cp, section=transec)
        if len(trans) > 0:
            trans = trans[0]
            samples = apply_transforms(samples, [trans])
            p1 = samples[params[0]]

            # We have transformed parameters, so automatically provide the
            # inverse transform for use in passing to waveform approximants
            class Thook(BaseTransform):
                name = subname
                _inputs = trans.outputs
                _outputs = trans.inputs
                p1name = params[0]
                sort = p1.argsort()
                p1sorted = p1[sort]
                def transform(self, maps):
                    idx = numpy.searchsorted(self.p1sorted, maps[self.p1name])
                    out = {p: samples[p][self.sort[idx]] for p in self.outputs}
                    return self.format_output(maps, out)
            global_transforms[Thook.name] = Thook
        return cls(params, samples)

__all__ = ['FixedSamples']
