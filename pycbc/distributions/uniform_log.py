# Copyright (C) 2017  Christopher M. Biwer
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
"""This modules provides classes for evaluating distributions whose logarithm
are uniform.
"""

import logging
import math

import numpy

from pycbc.distributions import bounded, uniform

logger = logging.getLogger("pycbc.distributions.uniform_log")


class UniformLog10(uniform.Uniform):
    r"""A uniform distribution on the log base 10 of the given parameters.
    The parameters are independent of each other. Instances of this class can
    be called like a function. By default, logpdf will be called.

    Parameters
    ----------
    \**params :
        The keyword arguments should provide the names of parameters and their
        corresponding bounds, as either tuples or a `boundaries.Bounds`
        instance.
    """

    name = "uniform_log10"

    def __init__(self, **params):
        super(UniformLog10, self).__init__(**params)
        self._norm = numpy.prod(
            [numpy.log10(bnd[1]) - numpy.log10(bnd[0]) for bnd in self._bounds.values()]
        )
        self._lognorm = numpy.log(self._norm)

    def _cdfinv_param(self, param, value):
        """Return the cdfinv for a single given parameter"""
        torch, reference = bounded._torch_module_and_reference((value,))
        if torch is not None:
            if not reference.is_floating_point() and not reference.is_complex():
                value = bounded._torch_as_tensor(value, reference)
            lower_bound = torch.as_tensor(
                math.log10(self._bounds[param][0]),
                dtype=value.dtype,
                device=value.device,
            )
            upper_bound = torch.as_tensor(
                math.log10(self._bounds[param][1]),
                dtype=value.dtype,
                device=value.device,
            )
            base = torch.as_tensor(10.0, dtype=value.dtype, device=value.device)
            return torch.pow(base, (upper_bound - lower_bound) * value + lower_bound)
        lower_bound = numpy.log10(self._bounds[param][0])
        upper_bound = numpy.log10(self._bounds[param][1])
        return 10.0 ** ((upper_bound - lower_bound) * value + lower_bound)

    def _pdf(self, **kwargs):
        """Returns the pdf at the given values. The keyword arguments must
        contain all of parameters in self's params. Unrecognized arguments are
        ignored.
        """
        torch, _ = bounded._torch_module_and_reference(kwargs.values())
        if torch is not None:
            return torch.exp(self._logpdf(**kwargs))
        if kwargs in self:
            vals = numpy.array(
                [numpy.log(10) * self._norm * kwargs[param] for param in kwargs.keys()]
            )
            return 1.0 / numpy.prod(vals)
        else:
            return 0.0

    def _logpdf(self, **kwargs):
        """Returns the log of the pdf at the given values. The keyword
        arguments must contain all of parameters in self's params. Unrecognized
        arguments are ignored.
        """
        contained = self.__contains__(kwargs)
        torch, reference = bounded._torch_module_and_reference(kwargs.values())
        if torch is not None:
            one = bounded._torch_as_tensor(1.0, reference)
            logpdf = bounded._torch_as_tensor(0.0, reference)
            scale = math.log(10) * self._norm
            for param in kwargs:
                value = kwargs[param]
                if not isinstance(value, torch.Tensor):
                    value = bounded._torch_as_tensor(value, reference)
                safe_value = torch.where(contained, value, one)
                logpdf = logpdf - torch.log(scale * safe_value)
            return bounded._torch_where(kwargs, contained, logpdf, -numpy.inf)
        if contained:
            return numpy.log(self._pdf(**kwargs))
        else:
            return -numpy.inf


__all__ = ["UniformLog10"]
