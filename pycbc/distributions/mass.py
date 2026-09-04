# Copyright (C) 2021 Yifan Wang
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

"""This modules provides classes for evaluating distributions for mchirp and
q (i.e., mass ratio) from uniform component mass.
"""
import logging
import numpy

from scipy.interpolate import CubicSpline, interp1d
from scipy.special import hyp2f1

from pycbc.distributions import power_law
from pycbc.distributions import bounded

logger = logging.getLogger('pycbc.distributions.mass')


class MchirpfromUniformMass1Mass2(power_law.UniformPowerLaw):
    r"""A distribution for chirp mass from uniform component mass +
    constraints given by chirp mass. This is a special case for UniformPowerLaw
    with index 1. For more details see UniformPowerLaw.

    The parameters (i.e. `**params`) are independent of each other. Instances
    of this class can be called like a function. By default, `logpdf` will be
    called, but this can be changed by setting the class's `__call__` method
    to its pdf method.

    Derivation for the probability density function:

    .. math::

        P(m_1,m_2)dm_1dm_2 = P(\mathcal{M}_c,q)d\mathcal{M}_cdq

    Where :math:`\mathcal{M}_c` is chirp mass and :math:`q` is mass ratio,
    :math:`m_1` and :math:`m_2` are component masses. The jacobian to transform
    chirp mass and mass ratio to component masses is

    .. math::

        \frac{\partial(m_1,m_2)}{\partial(\mathcal{M}_c,q)} = \
        \mathcal{M}_c \left(\frac{1+q}{q^3}\right)^{2/5}

    (https://github.com/gwastro/pycbc/blob/master/pycbc/transforms.py#L416.)

    Because :math:`P(m_1,m_2) = const`, then

    .. math::

        P(\mathcal{M}_c,q) = P(\mathcal{M}_c)P(q)\propto
        \mathcal{M}_c \left(\frac{1+q}{q^3}\right)^{2/5}`.

    Therefore,

    .. math::
        P(\mathcal{M}_c) \propto \mathcal{M}_c

    and

    .. math::
        P(q) \propto \left(\frac{1+q}{q^3}\right)^{2/5}

    Examples
    --------

    Generate 10000 random numbers from this distribution in [5,100]

    >>> from pycbc import distributions as dist
    >>> minmc = 5, maxmc = 100, size = 10000
    >>> mc = dist.MchirpfromUniformMass1Mass2(value=(minmc,maxmc)).rvs(size)

    The settings in the configuration file for pycbc_inference should be

    .. code-block:: ini

        [variable_params]
        mchirp =
        [prior-mchirp]
        name = mchirp_from_uniform_mass1_mass2
        min-mchirp = 10
        max-mchirp = 80

    Parameters
    ----------
    \**params :
        The keyword arguments should provide the names of parameters and their
        corresponding bounds, as either tuples or a `boundaries.Bounds`
        instance.
    """

    name = "mchirp_from_uniform_mass1_mass2"

    def __init__(self, dim=2, **params):
        super(MchirpfromUniformMass1Mass2, self).__init__(dim=2, **params)


class QfromUniformMass1Mass2(bounded.BoundedDist):
    r"""A distribution for mass ratio (i.e., q) from uniform component mass
    + constraints given by q.

    The parameters (i.e. `**params`) are independent of each other. Instances
    of this class can be called like a function. By default, `logpdf` will be
    called, but this can be changed by setting the class's `__call__` method
    to its pdf method.

    For mathematical derivation see the documentation above in the class
    `MchirpfromUniformMass1Mass2`.

    Parameters
    ----------
    \**params :
        The keyword arguments should provide the names of parameters and their
        corresponding bounds, as either tuples or a `boundaries.Bounds`
        instance.

    Examples
    --------

    Generate 10000 random numbers from this distribution in [1,8]

    >>> from pycbc import distributions as dist
    >>> minq = 1, maxq = 8, size = 10000
    >>> q = dist.QfromUniformMass1Mass2(value=(minq,maxq)).rvs(size)

    """

    name = 'q_from_uniform_mass1_mass2'

    def __init__(self, **params):
        super(QfromUniformMass1Mass2, self).__init__(**params)
        self._norm = 1.0
        self._lognorm = 0.0
        self._cdfinv_tables = {}
        for p in self._params:
            self._norm /= self._cdf_param(p, self._bounds[p][1]) - \
                self._cdf_param(p, self._bounds[p][0])
            q_array = numpy.linspace(
                self._bounds[p][0], self._bounds[p][1], num=1000,
                endpoint=True)
            cdf_array = self._cdf_param(p, q_array)
            coefficients = CubicSpline(
                cdf_array, q_array, bc_type='not-a-knot',
                extrapolate=False).c
            self._cdfinv_tables[p] = (
                cdf_array, q_array, coefficients)
        self._lognorm = numpy.log(self._norm)

    @property
    def norm(self):
        """float: The normalization of the multi-dimensional pdf."""
        return self._norm

    @property
    def lognorm(self):
        """float: The log of the normalization."""
        return self._lognorm

    def _pdf(self, **kwargs):
        """Returns the pdf at the given values. The keyword arguments must
        contain all of parameters in self's params. Unrecognized arguments are
        ignored.
        """
        for p in self._params:
            if p not in kwargs.keys():
                raise ValueError(
                    'Missing parameter {} to construct pdf.'.format(p))
        torch, _ = bounded._torch_module_and_reference(kwargs.values())
        if torch is not None:
            return torch.exp(self._logpdf(**kwargs))
        if kwargs in self:
            pdf = self._norm * \
                numpy.prod([(1.+kwargs[p])**(2./5)/kwargs[p]**(6./5)
                            for p in self._params])
            return float(pdf)
        else:
            return 0.0

    def _logpdf(self, **kwargs):
        """Returns the log of the pdf at the given values. The keyword
        arguments must contain all of parameters in self's params. Unrecognized
        arguments are ignored.
        """
        for p in self._params:
            if p not in kwargs.keys():
                raise ValueError(
                    'Missing parameter {} to construct logpdf.'.format(p))
        contained = self.__contains__(kwargs)
        torch, reference = bounded._torch_module_and_reference(
            kwargs.values()
        )
        if torch is not None:
            one = bounded._torch_as_tensor(1.0, reference)
            log_pdf = bounded._torch_as_tensor(self._lognorm, reference)
            for param in self._params:
                value = kwargs[param]
                if not isinstance(value, torch.Tensor):
                    value = bounded._torch_as_tensor(value, reference)
                safe_value = torch.where(contained, value, one)
                log_pdf = log_pdf + (
                    (2.0 / 5.0) * torch.log1p(safe_value)
                    - (6.0 / 5.0) * torch.log(safe_value)
                )
            return bounded._torch_where(
                kwargs, contained, log_pdf, -numpy.inf
            )
        if contained:
            return numpy.log(self._pdf(**kwargs))
        else:
            return -numpy.inf

    def _cdf_param(self, param, value):
        r""">>> from sympy import *
           >>> x = Symbol('x')
           >>> integrate((1+x)**(2/5)/x**(6/5))
           Output:
                             _
                      -0.2  |_  /-0.4, -0.2 |    I*pi\
                -5.0*x    * |   |           | x*e    |
                           2  1 \   0.8     |        /
        """
        if param in self._params:
            return -5. * value**(-1./5) * hyp2f1(-2./5, -1./5, 4./5, -value)
        else:
            raise ValueError('{} is not contructed yet.'.format(param))

    def _cdfinv_param(self, param, value):
        """Return the inverse cdf to map the unit interval to parameter bounds.
        Note that value should be uniform in [0,1]."""
        if param not in self._params:
            raise ValueError('{} is not contructed yet.'.format(param))
        torch, reference = bounded._torch_module_and_reference((value,))
        cdf_array, q_array, coefficients = self._cdfinv_tables[param]
        message = 'q_from_uniform_m1_m2 cdfinv requires input in [0,1].'
        if torch is not None:
            if reference.is_complex():
                raise TypeError(message)
            if not reference.is_floating_point():
                reference = reference.to(dtype=torch.get_default_dtype())
            invalid = (reference < 0) | (reference > 1)
            if bool(torch.any(invalid)):
                raise ValueError(message)
            knots = torch.as_tensor(
                cdf_array, dtype=reference.dtype, device=reference.device)
            coeffs = torch.as_tensor(
                coefficients,
                dtype=reference.dtype,
                device=reference.device,
            )
            target = (knots[-1] - knots[0]) * reference + knots[0]
            target = torch.clamp(target, min=knots[0], max=knots[-1])
            indices = torch.searchsorted(knots, target, right=True) - 1
            indices = indices.clamp(0, knots.numel() - 2)
            delta = target - knots[indices]
            return (
                (coeffs[0, indices] * delta + coeffs[1, indices]) * delta
                + coeffs[2, indices]
            ) * delta + coeffs[3, indices]

        value = numpy.asarray(value)
        if (value < 0).any() or (value > 1).any():
            raise ValueError(message)
        target = (cdf_array[-1] - cdf_array[0]) * value + cdf_array[0]
        target = numpy.clip(target, cdf_array[0], cdf_array[-1])
        q_invcdf_interp = interp1d(
            cdf_array, q_array, kind='cubic', bounds_error=True)
        return q_invcdf_interp(target)

    def rvs(self, size=1, param=None):
        """Gives a set of random values drawn from this distribution.

        Parameters
        ----------
        size : {1, int}
            The number of values to generate; default is 1.
        param : {None, string}
            If provided, will just return values for the given parameter.
            Otherwise, returns random values for each parameter.

        Returns
        -------
        structured array
            The random values in a numpy structured array. If a param was
            specified, the array will only have an element corresponding to the
            given parameter. Otherwise, the array will have an element for each
            parameter in self's params.
        """
        if param is not None:
            dtype = [(param, float)]
        else:
            dtype = [(p, float) for p in self.params]
        arr = numpy.zeros(size, dtype=dtype)
        for (p, _) in dtype:
            uniformcdfvalue = numpy.random.uniform(0, 1, size=size)
            arr[p] = self._cdfinv_param(p, uniformcdfvalue)
        return arr

    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a distribution based on a configuration file. The parameters
        for the distribution are retrieved from the section titled
        "[`section`-`variable_args`]" in the config file.

        Example:

        .. code-block:: ini

            [variable_params]
            q =
            [prior-q]
            name = q_from_uniform_mass1_mass2
            min-q = 1
            max-q = 8

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            ``VARARGS_DELIM``. These must appear in the "tag" part
            of the section header.

        Returns
        -------
        QfromUniformMass1Mass2
            A distribution instance from the pycbc.distributions.bounded
        module.
        """
        return super(QfromUniformMass1Mass2, cls).from_config(
            cp, section, variable_args, bounds_required=True)


__all__ = ["MchirpfromUniformMass1Mass2", "QfromUniformMass1Mass2"]
