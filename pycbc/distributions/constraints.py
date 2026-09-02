# Copyright (C) 2017 Christopher M. Biwer
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
This modules provides classes for evaluating multi-dimensional constraints.
"""
import logging
import re
import scipy.spatial
import numpy

from pycbc import transforms
from pycbc.io import record, HFile

logger = logging.getLogger('pycbc.distributions.constraints')


class Constraint(object):
    """Creates a constraint that evaluates to True if parameters obey
    the constraint and False if they do not.
    """
    name = "custom"

    def __init__(self, constraint_arg, static_args=None, transforms=None,
                 **kwargs):
        static_args = (
            {} if static_args is None
            else dict(sorted(
                static_args.items(), key=lambda x: len(x[0]), reverse=True))
            )
        for arg, val in static_args.items():
            swp = f"'{val}'" if isinstance(val, str) else str(val)
            # Substitute static arg name for value if it appears in the
            # constraint_arg string at the beginning of a word and is not
            # followed by an underscore or equals sign.
            # This ensures that static_args that are also kwargs in function calls are
            # handled correctly, i.e., the kwarg is not touched while its value is replaced
            # with the static_arg value.
            constraint_arg = re.sub(
                r'\b{}(?!\_|\=)'.format(arg), swp, constraint_arg)
        self.constraint_arg = constraint_arg
        self.transforms = transforms
        self._torch_code_cache = {}
        for kwarg in kwargs.keys():
            setattr(self, kwarg, kwargs[kwarg])

    def __call__(self, params):
        """Evaluates constraint.
        """

        if isinstance(params, dict) and type(self) is Constraint:
            input_names = set(params)
            out = transforms._evaluate_raw_torch_expression(
                self.constraint_arg,
                params,
                input_names,
                self._torch_code_cache,
                allowed_nodes=transforms._TORCH_CONSTRAINT_NODES,
            )
            if (
                out is transforms._TORCH_EXPRESSION_UNSUPPORTED
                and self.transforms
            ):
                transformed = transforms.apply_transforms(
                    params, self.transforms
                )
                input_names = set(transformed)
                out = transforms._evaluate_raw_torch_expression(
                    self.constraint_arg,
                    transformed,
                    input_names,
                    self._torch_code_cache,
                    allowed_nodes=transforms._TORCH_CONSTRAINT_NODES,
                )
            if out is not transforms._TORCH_EXPRESSION_UNSUPPORTED:
                reference = next(
                    (
                        value
                        for value in params.values()
                        if transforms._torch_module_for(value) is not None
                    ),
                    None,
                )
                torch = transforms._torch_module_for(reference)
                if not isinstance(out, torch.Tensor):
                    out = torch.as_tensor(out, device=reference.device)
                return out.to(dtype=torch.bool)

        if isinstance(params, dict):
            params = record.FieldArray.from_kwargs(**params)
        elif not isinstance(params, record.FieldArray):
            raise ValueError("params must be dict or FieldArray instance")

        try:
            out = self._constraint(params)
        except (NameError, AttributeError, TypeError):

            if self.transforms:
                params = transforms.apply_transforms(params, self.transforms)

            out = self._constraint(params)


        if isinstance(out, record.FieldArray):
            out = out.item() if params.size == 1 else out
        return out

    def _constraint(self, params):
        """ Evaluates constraint function.
        """
        return params[self.constraint_arg]


class SupernovaeConvexHull(Constraint):
    """Pre defined constraint for core-collapse waveforms that checks
    whether a given set of coefficients lie within the convex hull of
    the coefficients of the principal component basis vectors.
    """
    name = "supernovae_convex_hull"
    required_parameters = ["coeff_0", "coeff_1"]
    _torch_max_working_elements = 1 << 20

    def __init__(self, constraint_arg, transforms=None, **kwargs):
        super(SupernovaeConvexHull,
              self).__init__(constraint_arg, transforms=transforms, **kwargs)
        self._torch_hull_cache = {}

        if 'principal_components_file' in kwargs:
            pc_filename = kwargs['principal_components_file']
            hull_dimention = numpy.array(kwargs['hull_dimention'])
            self.hull_dimention = int(hull_dimention)
            pc_file = HFile(pc_filename, 'r')
            pc_coefficients = numpy.array(pc_file.get('coefficients'))
            pc_file.close()
            hull_points = []
            for dim in range(self.hull_dimention):
                hull_points.append(pc_coefficients[:, dim])
            hull_points = numpy.array(hull_points).T
            pc_coeffs_hull = scipy.spatial.Delaunay(hull_points)
            self._hull = pc_coeffs_hull
            self.required_parameters = [
                f"coeff_{dim}" for dim in range(self.hull_dimention)
            ]

    def __call__(self, params):
        """Evaluate tensor-valued coefficients without leaving Torch."""
        if isinstance(params, dict):
            reference = next(
                (
                    value for value in params.values()
                    if transforms._torch_module_for(value) is not None
                ),
                None,
            )
            torch = transforms._torch_module_for(reference)
            if torch is not None:
                try:
                    return self._torch_constraint(params, torch, reference)
                except (NameError, AttributeError, TypeError):
                    if not self.transforms:
                        raise
                    transformed = transforms.apply_transforms(
                        params, self.transforms
                    )
                    return self._torch_constraint(
                        transformed, torch, reference
                    )
        return super().__call__(params)

    def _torch_transform(self, torch, reference, dtype):
        """Return the cached Delaunay affine transforms on a device."""
        cache_key = (reference.device, dtype)
        try:
            return self._torch_hull_cache[cache_key]
        except KeyError:
            transform = torch.as_tensor(
                self._hull.transform,
                dtype=dtype,
                device=reference.device,
            )
            self._torch_hull_cache[cache_key] = transform
            return transform

    def _torch_constraint(self, params, torch, reference):
        """Test Delaunay barycentric coordinates in bounded chunks."""
        dtype = (
            reference.dtype
            if reference.is_floating_point()
            else torch.get_default_dtype()
        )
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32

        coefficients = []
        for dim in range(self.hull_dimention):
            value = params[f"coeff_{dim}"]
            if isinstance(value, torch.Tensor):
                if value.device != reference.device:
                    raise ValueError(
                        "convex-hull coefficients must share a Torch device"
                    )
                if value.is_complex():
                    raise TypeError(
                        "convex-hull coefficients must be real-valued"
                    )
            coefficients.append(torch.as_tensor(
                value, dtype=dtype, device=reference.device
            ))

        coefficients = torch.broadcast_tensors(*coefficients)
        output_shape = coefficients[0].shape
        points = torch.stack(coefficients, dim=-1).reshape(
            -1, self.hull_dimention
        )

        transform = self._torch_transform(
            torch, reference, points.dtype
        )
        matrices = transform[:, :self.hull_dimention, :]
        offsets = transform[:, self.hull_dimention, :]
        simplex_count = matrices.shape[0]
        working_size = max(
            1, simplex_count * self.hull_dimention
        )
        chunk_size = max(
            1, self._torch_max_working_elements // working_size
        )
        tolerance = 100 * torch.finfo(points.dtype).eps
        inside = torch.empty(
            len(points), dtype=torch.bool, device=reference.device
        )

        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            delta = points[start:stop, None, :] - offsets[None, :, :]
            barycentric = torch.einsum(
                "sij,nsj->nsi", matrices, delta
            )
            final_coordinate = 1 - barycentric.sum(dim=-1)
            simplex_inside = torch.logical_and(
                torch.all(barycentric >= -tolerance, dim=-1),
                final_coordinate >= -tolerance,
            )
            inside[start:stop] = torch.any(simplex_inside, dim=-1)
        return inside.reshape(output_shape)

    def _constraint(self, params):
        points = numpy.stack(
            [
                params[f"coeff_{dim}"]
                for dim in range(self.hull_dimention)
            ],
            axis=-1,
        )
        return self._hull.find_simplex(points) >= 0


# list of all constraints
constraints = {
    Constraint.name : Constraint,
    SupernovaeConvexHull.name : SupernovaeConvexHull,
}
