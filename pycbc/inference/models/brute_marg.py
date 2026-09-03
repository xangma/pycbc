# Copyright (C) 2020 Alex Nitz
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

"""This module provides model classes that do brute force marginalization
using at the likelihood level.
"""
import math
import logging
import numpy

from pycbc.pool import BroadcastPool as Pool
from scipy.special import logsumexp

from .gaussian_noise import BaseGaussianNoise
from .tools import _torch_tensor, draw_sample


def _stack_likelihood_values(values):
    """Stack likelihood scalars without detaching Torch values."""
    tensors = [_torch_tensor(value) for value in values]
    target = next((tensor for tensor in tensors if tensor is not None), None)
    if target is None:
        return numpy.asarray(values)

    import torch

    return torch.stack([
        tensor.to(device=target.device, dtype=target.dtype)
        if tensor is not None else
        torch.as_tensor(value, device=target.device, dtype=target.dtype)
        for value, tensor in zip(values, tensors)
    ])


def _argmax_index(values):
    """Return a host index after reducing on the input backend."""
    tensor = _torch_tensor(values)
    if tensor is not None:
        import torch

        return int(torch.argmax(tensor).item())
    return int(numpy.argmax(values))


def _host_scalar(value):
    """Convert one scalar result to its public host representation."""
    tensor = _torch_tensor(value)
    if tensor is not None:
        return tensor.item()
    if isinstance(value, numpy.generic):
        return value.item()
    return value


def _logmeanexp(values):
    """Marginalize a likelihood vector on its existing backend."""
    tensor = _torch_tensor(values)
    if tensor is not None:
        import torch

        result = torch.logsumexp(tensor, dim=0) - math.log(tensor.shape[0])
        return result.item()
    return float(logsumexp(values) - numpy.log(len(values)))


_model = None


class likelihood_wrapper(object):
    def __init__(self, model):
        global _model
        _model = model

    def __call__(self, params):
        global _model
        _model.update(**params)
        loglr = _model.loglr
        return loglr, _model.current_stats


class BruteParallelGaussianMarginalize(BaseGaussianNoise):
    name = "brute_parallel_gaussian_marginalize"

    def __init__(self, variable_params,
                 cores=10,
                 base_model=None,
                 marginalize_phase=None,
                 **kwds):
        super().__init__(variable_params, **kwds)

        from pycbc.inference.models import models
        self.model = models[base_model](variable_params, **kwds)

        self.call = likelihood_wrapper(self.model)

        # size of pool for each likelihood call
        self.pool = Pool(int(cores))

        # Only one for now, but can be easily extended
        self.phase = None
        if marginalize_phase:
            samples = int(marginalize_phase)
            self.phase = numpy.linspace(0, 2.0 * numpy.pi, samples)

    @property
    def _extra_stats(self):
        stats = self.model._extra_stats
        stats.append('maxl_phase')
        if 'maxl_loglr' not in stats:
            stats.append('maxl_loglr')
        return stats

    def _loglr(self):
        if self.phase is not None:
            params = []
            for p in self.phase:
                pref = self.current_params.copy()
                pref['coa_phase'] = p
                params.append(pref)
            vals = list(self.pool.map(self.call, params))
            loglr = _stack_likelihood_values([v[0] for v in vals])
            # get the maxl values
            if 'maxl_loglr' not in self.model._extra_stats:
                maxl_loglrs = loglr
            else:
                maxl_loglrs = _stack_likelihood_values(
                    [v[1]['maxl_loglr'] for v in vals]
                )
            maxidx = _argmax_index(maxl_loglrs)
            maxstats = vals[maxidx][1]
            maxphase = self.phase[maxidx]
            # set the stats
            for stat in maxstats:
                setattr(self._current_stats, stat, maxstats[stat])
            self._current_stats.maxl_phase = maxphase
            self._current_stats.maxl_loglr = _host_scalar(
                maxl_loglrs[maxidx]
            )
            # calculate the marginal loglr and return
            return _logmeanexp(loglr)


class BruteLISASkyModesMarginalize(BaseGaussianNoise):
    name = "brute_lisa_sky_modes_marginalize"

    def __init__(self, variable_params,
                 cores=1,
                 loop_polarization=False,
                 base_model=None,
                 **kwds):
        super().__init__(variable_params, **kwds)

        from pycbc.inference.models import models
        kwds.update(models[base_model].extra_args_from_config(
            kwds['config_object'],
            "model",
            skip_args=[])
        )

        self.model = models[base_model](variable_params, **kwds)

        self.call = likelihood_wrapper(self.model)

        # size of pool for each likelihood call
        if cores > 1:
            self.pool = Pool(int(cores))
            self.mapfunc = self.pool.map
        else:
            self.pool = None
            self.mapfunc = map

        # Do I explicitly check the polarization + pi/2 points
        # We could also add other arguments here, ie only check longitude
        # or latitude symmetry points.
        if loop_polarization:
            self.num_sky_modes = 16
        else:
            self.num_sky_modes = 8

        self.reconstruct_sky_points = False

    @property
    def _extra_stats(self):
        stats = self.model._extra_stats
        return stats

    def _loglr(self):
        params = []
        for sym_num in range(self.num_sky_modes):
            pref = self.current_params.copy()
            self._apply_sky_point_rotation(pref, sym_num)
            params.append(pref)

        vals = list(self.mapfunc(self.call, params))
        loglr = _stack_likelihood_values([v[0] for v in vals])

        if self.reconstruct_sky_points:
            return loglr

        marg_llr = _logmeanexp(loglr)

        # set the stats
        for sym_num in range(self.num_sky_modes):
            setattr(
                self._current_stats,
                f'llr_mode_{sym_num}',
                _host_scalar(loglr[sym_num]),
            )

        return marg_llr

    def _apply_sky_point_rotation(self, pref, sky_num):
        """ Apply the sky point rotation for mode sky_num to parameters pref
        """
        lambdal = pref['eclipticlongitude']
        beta = pref['eclipticlatitude']
        psi = pref['polarization']
        inc = pref['inclination']

        pol_num = sky_num // 8
        sky_num = sky_num % 8
        long_num = sky_num % 4
        lat_num = sky_num // 4

        # Apply latitude symmetry mode
        if lat_num:
            beta = - beta
            inc = numpy.pi - inc
            psi = numpy.pi - psi

        # Apply longitudonal symmetry mode
        lambdal = (lambdal + long_num * 0.5 * numpy.pi) % (2*numpy.pi)
        psi = (psi + long_num * 0.5 * numpy.pi) % (2*numpy.pi)

        # Apply additional polarization mode (shouldn't be needed)
        if pol_num:
            psi = psi + (math.pi / 2.)

        pref['eclipticlongitude'] = lambdal
        pref['eclipticlatitude'] = beta
        pref['polarization'] = psi
        pref['inclination'] = inc

    @classmethod
    def from_config(cls, cp, **kwargs):
        kwargs['config_object'] = cp
        return super(BruteLISASkyModesMarginalize, cls).from_config(
            cp,
            **kwargs
        )

    def reconstruct(self, seed=None):
        """ Reconstruct a point from unwrapping the 8-fold sky symmetry
        """
        if seed:
            numpy.random.seed(seed)
        rec = {}

        logging.info('Reconstruct LISA sky mode symmetry')
        self.reconstruct_sky_points = True
        loglr = self.loglr
        xl = draw_sample(loglr)
        logging.info('Found point %d', xl)
        # Undo rotations
        pref = self.current_params.copy()
        self._apply_sky_point_rotation(pref, xl)

        for val in ['polarization', 'eclipticlongitude', 'eclipticlatitude',
                    'inclination']:
            rec[val] = pref[val]
        rec['loglr'] = loglr[xl]
        rec['loglikelihood'] = self.lognl + rec['loglr']
        self.reconstruct_sky_points = False
        return self.model.reconstruct(seed=seed, rec=rec)
