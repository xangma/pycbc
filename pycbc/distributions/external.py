# Copyright (C) 2020 Alexander Nitz, 2022 Shichao Wu
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
This modules provides classes for evaluating PDF, logPDF, CDF and inverse CDF
from external arbitrary distributions, and drawing samples from them.
"""
import logging
import importlib
import numpy as np

import scipy.integrate as scipy_integrate
import scipy.interpolate as scipy_interpolate

from pycbc import VARARGS_DELIM
from pycbc.distributions import bounded

logger = logging.getLogger('pycbc.distributions.external')


class External(object):
    """ Distribution defined by external cdfinv and logpdf functions

    To add to an inference configuration file:

    .. code-block:: ini

        [prior-param1+param2]
        name = external
        module = custom_mod
        logpdf = custom_function_name
        cdfinv = custom_function_name2

    Parameters
    ----------
    params : list
        list of parameter names
    custom_mod : module
        module from which logpdf and cdfinv functions can be imported
    logpdf : function
        function which returns the logpdf
    cdfinv : function
        function which applies the invcdf

    Examples
    --------
    To instantate by hand and example of function format. You must provide
    the logpdf function, and you may either provide the rvs or cdfinv function.
    If the cdfinv is provided, but not the rvs, the random values will
    be calculated using the cdfinv function.

    >>> import numpy
    >>> params = ['x', 'y']
    >>> def logpdf(x=None, y=None):
    ...     p = numpy.ones(len(x))
    ...     return p
    >>>
    >>> def cdfinv(**kwds):
    ...     return kwds
    >>> e = External(['x', 'y'], logpdf, cdfinv=cdfinv)
    >>> e.rvs(size=10)
    """
    name = "external"

    def __init__(self, params=None, logpdf=None,
                 rvs=None, cdfinv=None, **kwds):
        self.params = params
        self.logpdf = logpdf
        self.cdfinv = cdfinv
        self._rvs = rvs

        if not (rvs or cdfinv):
            raise ValueError("Must provide either rvs or cdfinv")

    def rvs(self, size=1, **kwds):
        "Draw random value"
        if self._rvs:
            return self._rvs(size=size)
        samples = {param: np.random.uniform(0, 1, size=size)
                   for param in self.params}
        return self.cdfinv(**samples)

    def apply_boundary_conditions(self, **params):
        return params

    def __call__(self, **kwds):
        return self.logpdf(**kwds)

    @classmethod
    def from_config(cls, cp, section, variable_args):
        tag = variable_args
        params = variable_args.split(VARARGS_DELIM)
        modulestr = cp.get_opt_tag(section, 'module', tag)
        mod = importlib.import_module(modulestr)

        logpdfstr = cp.get_opt_tag(section, 'logpdf', tag)
        logpdf = getattr(mod, logpdfstr)

        cdfinv = rvs = None
        if cp.has_option_tag(section, 'cdfinv', tag):
            cdfinvstr = cp.get_opt_tag(section, 'cdfinv', tag)
            cdfinv = getattr(mod, cdfinvstr)

        if cp.has_option_tag(section, 'rvs', tag):
            rvsstr = cp.get_opt_tag(section, 'rvs', tag)
            rvs = getattr(mod, rvsstr)

        return cls(params=params, logpdf=logpdf, rvs=rvs, cdfinv=cdfinv)


class DistributionFunctionFromFile(External):
    r"""Evaluating PDF, logPDF, CDF and inverse CDF from the external
        density function.

    To add to an inference configuration file:

    .. code-block:: ini

        [prior-param1]
        name = external_func_fromfile
        file_path = spin.txt
        column_index = 1

    Parameters
    ----------
    params : list
        list of parameter names
    file_path: str
        The path of the external density function's .txt file.
    column_index: int
        The column index of the density distribution. By default, the first
        should be the values of a certain parameter, such as "mass", other
        columns should be the corresponding density values (as a function of
        that parameter). If you add the name of the parameter in the first
        row, please add the '#' at the beginning.
    \**kwargs :
        All other keyword args are passed to `scipy.integrate.quad` to control
        the numerical accuracy of the inverse CDF.
        If not be provided, will use the default values in `self.__init__`.

    Notes
    -----
    This class is different from `pycbc.distributions.arbitrary.FromFile`,
    which needs samples from the hdf file to construct the PDF by using KDE.
    This class reads in any continuous functions of the parameter.
    """
    name = "external_func_fromfile"

    def __init__(self, params=None, file_path=None,
                 column_index=None, **kwargs):
        super().__init__(cdfinv=self._cdfinv, logpdf=self.logpdf)
        self.params = params
        self.data = np.loadtxt(fname=file_path, unpack=True, comments='#')
        self.column_index = int(column_index)
        self.epsabs = kwargs.get('epsabs', 1.49e-05)
        self.epsrel = kwargs.get('epsrel', 1.49e-05)
        self.x_list = np.linspace(self.data[0][0], self.data[0][-1], 1000)
        self.interp = {'pdf': callable, 'cdf': callable, 'cdfinv': callable}
        self._pdf_x = None
        self._pdf_y = None
        self._cdf_x = None
        self._cdf_y = None
        self._cdfinv_x = None
        self._cdfinv_y = None
        self._torch_pdf_cache = {}
        self._torch_cdf_cache = {}
        self._torch_cdfinv_cache = {}
        if not file_path:
            raise ValueError("Must provide the path to density function file.")

    def logpdf(self, **kwargs):
        x = kwargs.pop(self.params[0])
        return self._logpdf(x, **kwargs)

    def _ensure_pdf_interpolator(self, **kwargs):
        """Build and cache the normalized PDF interpolation table."""
        if self.interp['pdf'] == callable:
            func_unnorm = scipy_interpolate.interp1d(
                self.data[0], self.data[self.column_index])
            norm_const = scipy_integrate.quad(
                func_unnorm, self.data[0][0], self.data[0][-1],
                epsabs=self.epsabs, epsrel=self.epsrel, limit=500,
                **kwargs)[0]
            self.interp['pdf'] = scipy_interpolate.interp1d(
                self.data[0], self.data[self.column_index]/norm_const,
                bounds_error=False, fill_value=0)
            # Use scipy's sorted interpolation knots so the Torch and scipy
            # evaluators have the same piecewise-linear representation.
            self._pdf_x = np.asarray(self.interp['pdf'].x)
            self._pdf_y = np.asarray(self.interp['pdf'].y)
            self._torch_pdf_cache = {}

    def _torch_pdf_tables(self, reference):
        """Return normalized interpolation knots on ``reference``'s device."""
        return self._torch_interpolation_tables("pdf", reference)

    def _torch_interpolation_tables(self, kind, reference):
        """Return an interpolation table on ``reference``'s device."""
        torch = bounded._torch_module_and_reference([reference])[0]
        dtype = (
            reference.dtype
            if reference.is_floating_point()
            else torch.get_default_dtype()
        )
        key = (reference.device.type, reference.device.index, dtype)
        cache = getattr(self, "_torch_{}_cache".format(kind))
        try:
            return cache[key]
        except KeyError:
            pass
        tables = (
            torch.as_tensor(
                getattr(self, "_{}_x".format(kind)),
                device=reference.device,
                dtype=dtype,
            ),
            torch.as_tensor(
                getattr(self, "_{}_y".format(kind)),
                device=reference.device,
                dtype=dtype,
            ),
        )
        cache[key] = tables
        return tables

    def _torch_linear_interpolate(self, kind, values, message):
        """Evaluate one cached interpolation table without leaving Torch."""
        torch, reference = bounded._torch_module_and_reference([values])
        if reference.is_complex():
            raise TypeError(message)
        x_knots, y_knots = self._torch_interpolation_tables(kind, reference)
        values = values.to(dtype=x_knots.dtype)
        invalid = (values < x_knots[0]) | (values > x_knots[-1])
        if bool(torch.any(invalid)):
            raise ValueError(message)
        indices = torch.searchsorted(x_knots, values).clamp(
            min=1, max=x_knots.numel() - 1
        )
        x_left = x_knots[indices - 1]
        x_right = x_knots[indices]
        y_left = y_knots[indices - 1]
        y_right = y_knots[indices]
        return y_left + (
            (values - x_left) * (y_right - y_left) / (x_right - x_left)
        )

    def _pdf(self, x010, **kwargs):
        """Calculate and interpolate the PDF by using the given density
        function, then return the corresponding value at the given x."""
        self._ensure_pdf_interpolator(**kwargs)
        torch, reference = bounded._torch_module_and_reference([x010])
        if torch is not None:
            x_knots, pdf_knots = self._torch_pdf_tables(reference)
            dtype = x_knots.dtype
            values = x010.to(dtype=dtype)
            indices = torch.searchsorted(x_knots, values).clamp(
                min=1, max=x_knots.numel() - 1
            )
            x_left = x_knots[indices - 1]
            x_right = x_knots[indices]
            pdf_left = pdf_knots[indices - 1]
            pdf_right = pdf_knots[indices]
            pdf_val = pdf_left + (
                (values - x_left)
                * (pdf_right - pdf_left)
                / (x_right - x_left)
            )
            inside = (values >= x_knots[0]) & (values <= x_knots[-1])
            return torch.where(inside, pdf_val, torch.zeros_like(pdf_val))
        pdf_val = np.float64(self.interp['pdf'](x010))
        return pdf_val

    def _logpdf(self, x010, **kwargs):
        """Calculate the logPDF by calling `pdf` function."""
        pdf_val = self._pdf(x010, **kwargs)
        torch = bounded._torch_module_and_reference([pdf_val])[0]
        if torch is not None:
            return pdf_val.log()
        return np.log(pdf_val)

    def _cdf(self, x, **kwargs):
        """Calculate and interpolate the CDF, then return the corresponding
        value at the given x."""
        self._ensure_cdf_interpolator(**kwargs)
        torch = bounded._torch_module_and_reference([x])[0]
        if torch is not None:
            message = "CDF input is outside the tabulated parameter range."
            return self._torch_linear_interpolate("cdf", x, message)
        cdf_val = np.float64(self.interp['cdf'](x))
        return cdf_val

    def _ensure_cdf_interpolator(self, **kwargs):
        """Build and cache the CDF interpolation table."""
        if self.interp['cdf'] == callable:
            cdf_list = []
            for x_val in self.x_list:
                cdf_x = scipy_integrate.quad(
                    self._pdf, self.data[0][0], x_val, epsabs=self.epsabs,
                    epsrel=self.epsrel, limit=500, **kwargs)[0]
                cdf_list.append(cdf_x)
            self.interp['cdf'] = \
                scipy_interpolate.interp1d(self.x_list, cdf_list)
            self._cdf_x = np.asarray(self.interp['cdf'].x)
            self._cdf_y = np.asarray(self.interp['cdf'].y)
            self._torch_cdf_cache = {}

    def _cdfinv(self, **kwargs):
        """Calculate and interpolate the inverse CDF, then return the
        corresponding parameter value at the given CDF value."""
        self._ensure_cdfinv_interpolator()
        value = kwargs[self.params[0]]
        torch = bounded._torch_module_and_reference([value])[0]
        if torch is not None:
            message = "inverse CDF input must lie in [0, 1]."
            return {self.params[0]: self._torch_linear_interpolate(
                "cdfinv", value, message
            )}
        cdfinv_val = {self.params[0]: np.float64(
            self.interp['cdfinv'](value))}
        return cdfinv_val

    def _ensure_cdfinv_interpolator(self):
        """Build and cache the inverse-CDF interpolation table."""
        if self.interp['cdfinv'] == callable:
            self._ensure_cdf_interpolator()
            self.interp['cdfinv'] = \
                scipy_interpolate.interp1d(self._cdf_y, self.x_list)
            self._cdfinv_x = np.asarray(self.interp['cdfinv'].x)
            self._cdfinv_y = np.asarray(self.interp['cdfinv'].y)
            self._torch_cdfinv_cache = {}

    @classmethod
    def from_config(cls, cp, section, variable_args):
        tag = variable_args
        params = variable_args.split(VARARGS_DELIM)
        file_path = cp.get_opt_tag(section, 'file_path', tag)
        column_index = cp.get_opt_tag(section, 'column_index', tag)
        return cls(params=params, file_path=file_path,
                   column_index=column_index)


__all__ = ['External', 'DistributionFunctionFromFile']
