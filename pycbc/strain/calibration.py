# Copyright (C) 2018 Colm Talbot
#
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
""" Functions for adding calibration factors to waveform templates.
"""

import numpy as np
from abc import (ABCMeta, abstractmethod)


_FITPACK_DEGREE = 3
_FITPACK_TOLERANCE = 0.001
_FITPACK_MAX_ITERATIONS = 20
_FITPACK_WARNINGS = {
    2: (
        "\nA theoretically impossible result was found during the iteration\n"
        "process for finding a smoothing spline with fp = s: s too small.\n"
        "There is an approximation returned but the corresponding weighted "
        "sum\nof squared residuals does not satisfy the condition "
        "abs(fp-s)/s < tol."
    ),
    3: (
        "\nThe maximal number of iterations maxit (set to 20 by the program)\n"
        "allowed for finding a smoothing spline with fp=s has been reached: "
        "s\ntoo small.\nThere is an approximation returned but the "
        "corresponding weighted sum\nof squared residuals does not satisfy "
        "the condition abs(fp-s)/s < tol."
    ),
}

# This is a specialization of the legacy Dierckx/FITPACK routines used by
# scipy.interpolate.UnivariateSpline (fpcurf, fpknot, fpdisc, and helpers).
# Calibration fixes k=3, w=1, s=m, and the bounding box to the data endpoints,
# so the unused general FITPACK arguments are deliberately not exposed here.


def _fitpack_basis(knots, value, span, degree=_FITPACK_DEGREE):
    """Return FITPACK's nonzero B-spline basis values at one point."""
    basis = [0.0] * (degree + 1)
    basis[0] = 1.0
    for level in range(1, degree + 1):
        previous = basis[:level]
        basis[0] = 0.0
        for index in range(level):
            right_index = span + index + 1
            left_index = right_index - level
            denominator = knots[right_index] - knots[left_index]
            if denominator == 0.0:
                basis[index + 1] = 0.0
                continue
            factor = previous[index] / denominator
            basis[index] += factor * (knots[right_index] - value)
            basis[index + 1] = factor * (value - knots[left_index])
    return basis


def _fitpack_givens(pivot, diagonal):
    """Apply FITPACK's stable construction of a Givens rotation."""
    pivot_magnitude = abs(pivot)
    if pivot_magnitude >= diagonal:
        new_diagonal = pivot_magnitude * (
            1.0 + (diagonal / pivot) ** 2
        ) ** 0.5
    else:
        new_diagonal = diagonal * (
            1.0 + (pivot / diagonal) ** 2
        ) ** 0.5
    return (
        new_diagonal,
        diagonal / new_diagonal,
        pivot / new_diagonal,
    )


def _fitpack_tensor_givens(pivot, diagonal):
    """Torch equivalent of ``fpgivs``, retaining the scalar graph."""
    import torch

    if not torch.is_tensor(diagonal):
        diagonal = pivot.new_tensor(diagonal)
    pivot_magnitude = abs(pivot)
    if bool((pivot_magnitude >= diagonal).detach().cpu()):
        new_diagonal = pivot_magnitude * torch.sqrt(
            1.0 + (diagonal / pivot) ** 2
        )
    else:
        new_diagonal = diagonal * torch.sqrt(
            1.0 + (pivot / diagonal) ** 2
        )
    return (
        new_diagonal,
        diagonal / new_diagonal,
        pivot / new_diagonal,
    )


def _fitpack_rotate(cosine, sine, first, second):
    """Apply one FITPACK Givens rotation without in-place tensor writes."""
    old_first = first
    old_second = second
    return (
        cosine * old_first - sine * old_second,
        cosine * old_second + sine * old_first,
    )


def _fitpack_backsolve(matrix, right_hand_side, bandwidth):
    """Solve FITPACK's packed upper-triangular band system."""
    size = len(right_hand_side)
    coefficients = [None] * size
    coefficients[-1] = right_hand_side[-1] / matrix[-1][0]
    for row in range(size - 2, -1, -1):
        value = right_hand_side[row]
        count = min(bandwidth - 1, size - row - 1)
        for offset in range(1, count + 1):
            value = value - coefficients[row + offset] * matrix[row][offset]
        coefficients[row] = value / matrix[row][0]
    return coefficients


def _fitpack_least_squares(x, y, knots, degree=_FITPACK_DEGREE):
    """Build and reduce FITPACK's least-squares observation matrix."""
    order = degree + 1
    coefficient_count = len(knots) - order
    matrix = [[0.0] * order for _ in range(coefficient_count)]
    zero = y[0] * 0.0
    right_hand_side = [zero for _ in range(coefficient_count)]
    basis_rows = []
    coefficient_starts = []
    residual_sum = zero
    span = degree

    for point_index, point in enumerate(x):
        while (
            span < coefficient_count - 1
            and point >= knots[span + 1]
        ):
            span += 1
        basis = _fitpack_basis(knots, point, span, degree)
        # FITPACK stores q before rotating its separate h work row.
        basis_rows.append(basis.copy())
        start = span - degree
        coefficient_starts.append(start)
        value = y[point_index]

        for basis_index in range(order):
            row = start + basis_index
            pivot = basis[basis_index]
            if pivot == 0.0:
                continue
            diagonal, cosine, sine = _fitpack_givens(
                pivot, matrix[row][0]
            )
            matrix[row][0] = diagonal
            value, right_hand_side[row] = _fitpack_rotate(
                cosine, sine, value, right_hand_side[row]
            )
            for trailing in range(basis_index + 1, order):
                packed_index = trailing - basis_index
                basis[trailing], matrix[row][packed_index] = (
                    _fitpack_rotate(
                        cosine,
                        sine,
                        basis[trailing],
                        matrix[row][packed_index],
                    )
                )
        residual_sum = residual_sum + value * value

    coefficients = _fitpack_backsolve(
        matrix, right_hand_side, order
    )
    return (
        matrix,
        right_hand_side,
        basis_rows,
        coefficient_starts,
        coefficients,
        residual_sum,
    )


def _fitpack_residuals(y, basis_rows, starts, coefficients):
    """Evaluate squared residuals in FITPACK's operation order."""
    residuals = []
    for value, basis, start in zip(y, basis_rows, starts):
        fitted = value * 0.0
        for offset in range(_FITPACK_DEGREE + 1):
            fitted = fitted + coefficients[start + offset] * basis[offset]
        residual = fitted - value
        residuals.append(residual * residual)
    return residuals


def _fitpack_interval_residuals(x, knots, residuals):
    """Partition residuals exactly as ``fpcurf`` does before ``fpknot``."""
    degree = _FITPACK_DEGREE
    order = degree + 1
    coefficient_count = len(knots) - order
    interval_count = len(knots) - 2 * order + 1
    interval_residuals = []
    partial = residuals[0] * 0.0
    boundary = degree + 1

    for point, residual in zip(x, residuals):
        crossed = False
        if not (
            point < knots[boundary]
            or boundary + 1 > coefficient_count
        ):
            crossed = True
            boundary += 1
        partial = partial + residual
        if crossed:
            half = residual * 0.5
            interval_residuals.append(partial - half)
            partial = half
    interval_residuals.append(partial)
    if len(interval_residuals) != interval_count:
        raise RuntimeError("FITPACK residual interval bookkeeping failed")
    return interval_residuals


def _fitpack_insert_knot(x, knots, interval_residuals, point_counts):
    """Port FITPACK's ``fpknot`` update, including its first-tie rule."""
    maximum = 0.0
    selected = None
    beginning = 1
    selected_beginning = None
    selected_count = None
    for interval, point_count in enumerate(point_counts):
        residual = float(interval_residuals[interval].detach().cpu())
        if point_count != 0 and residual > maximum:
            maximum = residual
            selected = interval
            selected_beginning = beginning
            selected_count = point_count
        beginning += point_count + 1

    if selected is None:
        # FITPACK still increments n on this theoretically impossible path.
        raise RuntimeError("FITPACK could not locate an interval for a knot")

    halfway = selected_count // 2 + 1
    point_index = selected_beginning + halfway - 1
    left_count = halfway - 1
    right_count = selected_count - halfway
    old_residual = interval_residuals[selected]

    knots.insert(selected + _FITPACK_DEGREE + 1, x[point_index])
    point_counts[selected] = left_count
    point_counts.insert(selected + 1, right_count)
    interval_residuals[selected] = (
        old_residual * left_count / selected_count
    )
    interval_residuals.insert(
        selected + 1,
        old_residual * right_count / selected_count,
    )


def _fitpack_discontinuities(knots):
    """Return FITPACK's cubic derivative-discontinuity rows."""
    degree = _FITPACK_DEGREE
    order = degree + 1
    width = order + 1
    knot_count = len(knots)
    coefficient_count = knot_count - order
    interval_count = coefficient_count - degree
    factor = interval_count / (
        knots[coefficient_count] - knots[order - 1]
    )
    rows = []

    # This deliberately follows fpdisc's multiplication order.
    for knot_index in range(width, coefficient_count + 1):
        work = [0.0] * (2 * order)
        row_number = knot_index - order
        for index in range(1, order + 1):
            far_index = index + order
            right_index = knot_index + index
            left_index = right_index - width
            work[index - 1] = (
                knots[knot_index - 1] - knots[left_index - 1]
            )
            work[far_index - 1] = (
                knots[knot_index - 1] - knots[right_index - 1]
            )
        row = []
        left = row_number
        for index in range(1, width + 1):
            work_index = index
            product = work[index - 1]
            for _ in range(degree):
                work_index += 1
                product = product * work[work_index - 1] * factor
            right = left + order
            row.append(
                (knots[right - 1] - knots[left - 1]) / product
            )
            left += 1
        rows.append(row)
    return rows


def _fitpack_smoothing_coefficients(
        x, y, knots, matrix, right_hand_side, basis_rows, starts,
        polynomial_residual, least_squares_residual, smoothing):
    """Solve FITPACK's fixed-topology smoothing problem on the Torch device."""
    order = _FITPACK_DEGREE + 1
    width = order + 1
    coefficient_count = len(knots) - order
    discontinuities = _fitpack_discontinuities(knots)
    discontinuity_count = len(discontinuities)
    tolerance = _FITPACK_TOLERANCE * smoothing
    zero = y[0] * 0.0

    p1 = zero
    f1 = polynomial_residual - smoothing
    p3 = zero - 1.0
    f3 = least_squares_residual - smoothing
    initial_p = coefficient_count / sum(
        matrix[row][0] for row in range(coefficient_count)
    )
    p = zero + initial_p
    lower_bracketed = False
    upper_bracketed = False
    coefficients = None

    for iteration in range(_FITPACK_MAX_ITERATIONS):
        inverse_p = 1.0 / p
        augmented = [
            [zero + matrix[row][column]
             if column < order else zero
             for column in range(width)]
            for row in range(coefficient_count)
        ]
        augmented_rhs = [value + zero for value in right_hand_side]

        for row_index, discontinuity in enumerate(discontinuities):
            work = [zero + value * inverse_p for value in discontinuity]
            value = zero
            for matrix_row in range(row_index, coefficient_count):
                pivot = work[0]
                diagonal, cosine, sine = _fitpack_tensor_givens(
                    pivot, augmented[matrix_row][0]
                )
                augmented[matrix_row][0] = diagonal
                value, augmented_rhs[matrix_row] = _fitpack_rotate(
                    cosine, sine, value, augmented_rhs[matrix_row]
                )
                if matrix_row == coefficient_count - 1:
                    break
                trailing_count = order
                if matrix_row + 1 > discontinuity_count:
                    trailing_count = coefficient_count - matrix_row - 1
                for trailing in range(1, trailing_count + 1):
                    work[trailing], augmented[matrix_row][trailing] = (
                        _fitpack_rotate(
                            cosine,
                            sine,
                            work[trailing],
                            augmented[matrix_row][trailing],
                        )
                    )
                    work[trailing - 1] = work[trailing]
                work[trailing_count] = zero

        coefficients = _fitpack_backsolve(
            augmented, augmented_rhs, width
        )
        residual = sum(_fitpack_residuals(
            y, basis_rows, starts, coefficients
        ), zero)
        difference = residual - smoothing
        if abs(float(difference.detach().cpu())) < tolerance:
            return coefficients, 0
        if iteration == _FITPACK_MAX_ITERATIONS - 1:
            return coefficients, 3

        p2 = p
        f2 = difference
        if not upper_bracketed:
            if float((f2 - f3).detach().cpu()) <= tolerance:
                p3 = p2
                f3 = f2
                p = p * 0.04
                if bool((p <= p1).detach().cpu()):
                    p = p1 * 0.9 + p2 * 0.1
                continue
            if float(f2.detach().cpu()) < 0.0:
                upper_bracketed = True

        if not lower_bracketed:
            if float((f1 - f2).detach().cpu()) <= tolerance:
                p1 = p2
                f1 = f2
                p = p / 0.04
                if float(p3.detach().cpu()) < 0.0:
                    continue
                if bool((p >= p3).detach().cpu()):
                    p = p2 * 0.1 + p3 * 0.9
                continue
            if float(f2.detach().cpu()) > 0.0:
                lower_bracketed = True

        if (
            bool((f2 >= f1).detach().cpu())
            or bool((f2 <= f3).detach().cpu())
        ):
            return coefficients, 2

        if float(p3.detach().cpu()) <= 0.0:
            p = (
                p1 * (f1 - f3) * f2
                - p2 * (f2 - f3) * f1
            ) / ((f1 - f2) * f3)
        else:
            h1 = f1 * (f2 - f3)
            h2 = f2 * (f3 - f1)
            h3 = f3 * (f1 - f2)
            p = -(
                p1 * p2 * h3 + p2 * p3 * h1 + p3 * p1 * h2
            ) / (p1 * h1 + p2 * h2 + p3 * h3)
        if float(f2.detach().cpu()) < 0.0:
            p3 = p2
            f3 = f2
        else:
            p1 = p2
            f1 = f2

    return coefficients, 3


def _torch_parameter_vector(parameters, reference):
    """Stack calibration parameters on the strain device without detaching."""
    import torch

    dtype = torch.float32 if reference.device.type == 'mps' else torch.float64
    values = []
    for parameter in parameters:
        value = torch.as_tensor(
            parameter, device=reference.device, dtype=dtype
        )
        if value.numel() != 1:
            raise ValueError("calibration spline parameters must be scalars")
        values.append(value.reshape(()))
    return torch.stack(values)


def _torch_fitpack_spline(spline_points, parameters, reference):
    """Fit the legacy default cubic smoothing spline without SciPy.

    FITPACK's singular-system behavior for repeated or non-finite abscissae is
    implementation-defined.  The public calibration path retains SciPy for
    those legacy edge cases and calls this native implementation only for
    finite, strictly increasing nodes.
    """
    import torch

    x = np.asarray(spline_points, dtype=np.float64)
    if x.ndim != 1 or len(x) != len(parameters):
        raise ValueError("x and y should have a same length")
    if len(x) < _FITPACK_DEGREE + 1:
        raise ValueError("m>k failed for hidden m: fpcurf0:m=0")
    if not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0.0):
        raise ValueError("x must be increasing if s > 0")
    y = _torch_parameter_vector(parameters, reference)
    smoothing = float(len(x))
    tolerance = _FITPACK_TOLERANCE * smoothing
    order = _FITPACK_DEGREE + 1
    minimum_knot_count = 2 * order
    maximum_knot_count = len(x) + order
    initial_capacity = max(len(x) // 2, minimum_knot_count)
    capacity = initial_capacity
    knots = [float(x[0])] * order + [float(x[-1])] * order
    point_counts = [len(x) - 2]
    previous_residual = 0.0
    knots_to_add = 0
    polynomial_residual = None
    status = -2

    for _ in range(2 * len(x) + 2):
        if len(knots) == minimum_knot_count:
            status = -2
        (
            matrix,
            right_hand_side,
            basis_rows,
            starts,
            coefficients,
            residual,
        ) = _fitpack_least_squares(x, y, knots)
        if status == -2:
            polynomial_residual = residual
        fit_is_finite = torch.isfinite(residual) & torch.stack([
            torch.isfinite(value) for value in coefficients
        ]).all()
        if not bool(fit_is_finite.detach().cpu()):
            break
        difference = residual - smoothing
        if abs(float(difference.detach().cpu())) < tolerance:
            break
        if float(difference.detach().cpu()) < 0.0:
            if status != -2:
                coefficients, smoothing_status = (
                    _fitpack_smoothing_coefficients(
                        x,
                        y,
                        knots,
                        matrix,
                        right_hand_side,
                        basis_rows,
                        starts,
                        polynomial_residual,
                        residual,
                        smoothing,
                    )
                )
                if smoothing_status:
                    import warnings

                    warnings.warn(
                        _FITPACK_WARNINGS[smoothing_status], stacklevel=3
                    )
            break

        if len(knots) == maximum_knot_count:
            break
        if len(knots) == capacity:
            if capacity == maximum_knot_count:
                break
            # UnivariateSpline resumes the same FITPACK state with more room.
            capacity = maximum_knot_count
            status = 1

        if status != 0:
            knots_to_add = 1
            status = 0
        else:
            proposed = knots_to_add * 2
            improvement = previous_residual - float(residual.detach().cpu())
            if improvement > tolerance:
                proposed = int(
                    knots_to_add
                    * float(difference.detach().cpu())
                    / improvement
                )
            knots_to_add = min(
                knots_to_add * 2,
                max(proposed, knots_to_add // 2, 1),
            )
        previous_residual = float(residual.detach().cpu())
        interval_residuals = _fitpack_interval_residuals(
            x,
            knots,
            _fitpack_residuals(y, basis_rows, starts, coefficients),
        )

        for _ in range(knots_to_add):
            _fitpack_insert_knot(
                x, knots, interval_residuals, point_counts
            )
            if len(knots) == maximum_knot_count:
                knots = (
                    [float(x[0])] * order
                    + [float(value) for value in x[2:-2]]
                    + [float(x[-1])] * order
                )
                break
            if len(knots) == capacity:
                break
    else:
        raise RuntimeError("Torch FITPACK knot selection did not converge")

    dtype = y.dtype
    device = y.device
    return (
        torch.as_tensor(knots, dtype=dtype, device=device),
        torch.stack(coefficients),
    )


def _evaluate_torch_spline(knots, coefficients, samples, degree):
    """Evaluate a tensor-coefficient B-spline with de Boor's algorithm."""
    import torch

    flat_samples = samples.reshape(-1)
    spans = torch.searchsorted(knots, flat_samples, right=True) - 1
    spans = spans.clamp(min=degree, max=len(coefficients) - 1)
    values = [
        coefficients[spans - degree + offset]
        for offset in range(degree + 1)
    ]
    for level in range(1, degree + 1):
        for index in range(degree, level - 1, -1):
            left = knots[spans - degree + index]
            right = knots[spans + 1 + index - level]
            weight = (flat_samples - left) / (right - left)
            values[index] = (
                (1.0 - weight) * values[index - 1]
                + weight * values[index]
            )
    return values[degree].reshape(samples.shape)


def _evaluate_spline(spline, sample_frequencies):
    """Evaluate a SciPy spline without copying a Torch grid to the host."""
    tensor = getattr(
        getattr(sample_frequencies, '_data', None), 'tensor', None
    )
    if tensor is None:
        return spline(sample_frequencies.numpy())

    import torch

    base_knots = np.asarray(spline.get_knots())
    coefficients = np.asarray(spline.get_coeffs())
    degree = len(coefficients) - len(base_knots) + 1
    knots = np.concatenate((
        np.repeat(base_knots[0], degree),
        base_knots,
        np.repeat(base_knots[-1], degree),
    ))

    samples = tensor
    knots = torch.as_tensor(
        knots, dtype=samples.dtype, device=samples.device
    )
    coefficients = torch.as_tensor(
        coefficients, dtype=samples.dtype, device=samples.device
    )
    return _evaluate_torch_spline(
        knots, coefficients, samples, degree
    )


def _multiply_frequency_series(strain, correction):
    """Apply a correction while preserving the strain storage scheme."""
    tensor = getattr(getattr(strain, '_data', None), 'tensor', None)
    if tensor is None:
        return strain * correction

    from pycbc.types.array_torch import TorchArrayData
    return strain._return(TorchArrayData(tensor * correction))


def _apply_spline_calibration(strain, spline_points,
                              amplitude_parameters, phase_parameters):
    """Fit and apply the standard amplitude and phase calibration splines.

    Torch-backed strain uses the FITPACK-equivalent fit above, keeping fit
    arrays and parameter gradients on its device. Adaptive knot topology is
    discrete, so its scalar branch decisions synchronize in eager execution.
    Non-Torch strain retains SciPy's ``UnivariateSpline`` implementation.
    Torch-backed strain also retains that path for repeated or non-finite
    nodes, whose singular FITPACK behavior is not a stable tensor operation.
    """
    frequencies = strain.sample_frequencies
    tensor = getattr(getattr(frequencies, '_data', None), 'tensor', None)
    spline_points_array = np.asarray(spline_points, dtype=np.float64)
    native_nodes_supported = (
        spline_points_array.ndim == 1
        and np.all(np.isfinite(spline_points_array))
        and np.all(np.diff(spline_points_array) > 0.0)
    )
    if tensor is None or not native_nodes_supported:
        from scipy.interpolate import UnivariateSpline

        amplitude_spline = UnivariateSpline(
            spline_points, amplitude_parameters
        )
        delta_amplitude = _evaluate_spline(amplitude_spline, frequencies)
        phase_spline = UnivariateSpline(spline_points, phase_parameters)
        delta_phase = _evaluate_spline(phase_spline, frequencies)
    else:
        amplitude_knots, amplitude_coefficients = _torch_fitpack_spline(
            spline_points, amplitude_parameters, tensor
        )
        delta_amplitude = _evaluate_torch_spline(
            amplitude_knots,
            amplitude_coefficients,
            tensor,
            _FITPACK_DEGREE,
        )
        phase_knots, phase_coefficients = _torch_fitpack_spline(
            spline_points, phase_parameters, tensor
        )
        delta_phase = _evaluate_torch_spline(
            phase_knots,
            phase_coefficients,
            tensor,
            _FITPACK_DEGREE,
        )

    correction = (
        (1.0 + delta_amplitude)
        * (2.0 + 1j * delta_phase)
        / (2.0 - 1j * delta_phase)
    )
    return _multiply_frequency_series(strain, correction)


class Recalibrate(metaclass=ABCMeta):
    name = None

    def __init__(self, ifo_name):
        self.ifo_name = ifo_name
        self.params = dict()

    @abstractmethod
    def apply_calibration(self, strain):
        """Apply calibration model

        This method should be overwritten by subclasses

        Parameters
        ----------
        strain : FrequencySeries
            The strain to be recalibrated.

        Return
        ------
        strain_adjusted : FrequencySeries
            The recalibrated strain.
        """
        return

    def map_to_adjust(self, strain, prefix='recalib_', **params):
        """Map an input dictionary of sampling parameters to the
        adjust_strain function by filtering the dictionary for the
        calibration parameters, then calling adjust_strain.

        Parameters
        ----------
        strain : FrequencySeries
            The strain to be recalibrated.
        prefix: str
            Prefix for calibration parameter names
        params : dict
            Dictionary of sampling parameters which includes
            calibration parameters.
        Return
        ------
        strain_adjusted : FrequencySeries
            The recalibrated strain.
        """

        self.params.update({
            key[len(prefix):]: params[key]
            for key in params if prefix in key and self.ifo_name in key})

        strain_adjusted = self.apply_calibration(strain)

        return strain_adjusted

    @classmethod
    def from_config(cls, cp, ifo, section):
        """Read a config file to get calibration options and transfer
        functions which will be used to intialize the model.

        Parameters
        ----------
        cp : WorkflowConfigParser
            An open config file.
        ifo : string
            The detector (H1, L1) for which the calibration model will
            be loaded.
        section : string
            The section name in the config file from which to retrieve
            the calibration options.
        Return
        ------
        instance
            An instance of the class.
        """
        all_params = dict(cp.items(section))
        params = {key[len(ifo)+1:]: all_params[key]
                  for key in all_params if ifo.lower() in key}
        model = params.pop('model')
        params['ifo_name'] = ifo.lower()

        return all_models[model](**params)


class CubicSpline(Recalibrate):
    name = 'cubic_spline'

    def __init__(self, minimum_frequency, maximum_frequency, n_points,
                 ifo_name):
        """
        Cubic spline recalibration

        see https://dcc.ligo.org/LIGO-T1400682/public

        This assumes the spline points follow
        np.logspace(np.log(minimum_frequency), np.log(maximum_frequency),
                    n_points)

        Parameters
        ----------
        minimum_frequency: float
            minimum frequency of spline points
        maximum_frequency: float
            maximum frequency of spline points
        n_points: int
            number of spline points
        """
        Recalibrate.__init__(self, ifo_name=ifo_name)
        minimum_frequency = float(minimum_frequency)
        maximum_frequency = float(maximum_frequency)
        n_points = int(n_points)
        if n_points < 4:
            raise ValueError(
                'Use at least 4 spline points for calibration model')
        self.n_points = n_points
        self.spline_points = np.logspace(np.log10(minimum_frequency),
                                         np.log10(maximum_frequency), n_points)

    def apply_calibration(self, strain):
        """Apply calibration model

        This applies cubic spline calibration to the strain.

        Parameters
        ----------
        strain : FrequencySeries
            The strain to be recalibrated.

        Return
        ------
        strain_adjusted : FrequencySeries
            The recalibrated strain.
        """
        amplitude_parameters =\
            [self.params['amplitude_{}_{}'.format(self.ifo_name, ii)]
             for ii in range(self.n_points)]
        phase_parameters =\
            [self.params['phase_{}_{}'.format(self.ifo_name, ii)]
             for ii in range(self.n_points)]
        return _apply_spline_calibration(
            strain, self.spline_points,
            amplitude_parameters, phase_parameters
        )


all_models = {
    CubicSpline.name: CubicSpline
}
