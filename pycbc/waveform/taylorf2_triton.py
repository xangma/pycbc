# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Optional fused CUDA evaluation of already prepared TaylorF2 batches.

The Torch batch implementation owns all physical validation, PN coefficients,
reference phases, amplitude normalization, and support metadata. This module
only evaluates those coefficients on the frequency grid. All arithmetic is
double precision, including the libdevice transcendental functions.
"""

import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    _HAS_TRITON = False
else:
    _HAS_TRITON = True


if _HAS_TRITON:

    @triton.jit
    def _taylorf2_kernel(
        plus_ptr,
        cross_ptr,
        coeff_ptr,
        coeff_log_ptr,
        coeff_log_sq_ptr,
        row_params_ptr,
        first_bins_ptr,
        end_bins_ptr,
        BATCH: tl.constexpr,
        LENGTH: tl.constexpr,
        DELTA_F: tl.constexpr,
        TIME_SHIFT: tl.constexpr,
        PI_OVER_FOUR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(1)
        bins = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        in_output = bins < LENGTH
        first = tl.load(first_bins_ptr + row)
        end = tl.load(end_bins_ptr + row)
        active = in_output & (bins >= first) & (bins < end)

        pi_mass = tl.load(row_params_ptr + row)
        reference_phase = tl.load(row_params_ptr + BATCH + row)
        amplitude_factor = tl.load(row_params_ptr + 2 * BATCH + row)
        coa_phase = tl.load(row_params_ptr + 3 * BATCH + row)
        cos_inclination = tl.load(row_params_ptr + 4 * BATCH + row)
        cos_nodes = tl.load(row_params_ptr + 5 * BATCH + row)
        sin_nodes = tl.load(row_params_ptr + 6 * BATCH + row)

        # Inactive bins use a positive frequency to avoid log(0) and division
        # by zero. Their final output is explicitly zero, including row tails.
        frequency = tl.where(active, bins, first).to(tl.float64) * tl.full(
            (), DELTA_F, tl.float64
        )
        velocity = libdevice.pow(
            pi_mass * frequency, tl.full((), 1.0 / 3.0, tl.float64)
        )
        log_velocity = libdevice.log(velocity)
        log_velocity_sq = log_velocity * log_velocity
        offset = 15 * BATCH + row
        phase = (
            tl.load(coeff_ptr + offset)
            + tl.load(coeff_log_ptr + offset) * log_velocity
            + tl.load(coeff_log_sq_ptr + offset) * log_velocity_sq
        )
        for index in tl.static_range(14, -1, -1):
            offset = index * BATCH + row
            term = (
                tl.load(coeff_ptr + offset)
                + tl.load(coeff_log_ptr + offset) * log_velocity
                + tl.load(coeff_log_sq_ptr + offset) * log_velocity_sq
            )
            # Match the CUDA torch.addcmul Horner step; other expressions
            # retain their explicit operation order (enable_fp_fusion=False).
            phase = tl.fma(phase, velocity, term)
        velocity_sq = velocity * velocity
        phase = phase / (velocity_sq * velocity_sq * velocity)
        phase = (
            phase
            + tl.full((), TIME_SHIFT, tl.float64) * frequency
            - 2.0 * coa_phase
            - reference_phase
            - tl.full((), PI_OVER_FOUR, tl.float64)
        )
        amplitude = amplitude_factor * libdevice.pow(velocity, -3.5)
        sample_real = amplitude * libdevice.cos(phase)
        sample_imag = -amplitude * libdevice.sin(phase)

        plus_factor = 0.5 * (1.0 + cos_inclination * cos_inclination)
        plus_real = sample_real * plus_factor
        plus_imag = sample_imag * plus_factor
        cross_real = sample_imag * cos_inclination
        cross_imag = -sample_real * cos_inclination
        rotated_plus_real = cos_nodes * plus_real + sin_nodes * cross_real
        rotated_plus_imag = cos_nodes * plus_imag + sin_nodes * cross_imag
        rotated_cross_real = cos_nodes * cross_real - sin_nodes * plus_real
        rotated_cross_imag = cos_nodes * cross_imag - sin_nodes * plus_imag

        # Complex128 tensors are exposed as interleaved float64 storage.
        output_offset = 2 * (row * LENGTH + bins)
        tl.store(
            plus_ptr + output_offset,
            tl.where(active, rotated_plus_real, 0.0),
            mask=in_output,
        )
        tl.store(
            plus_ptr + output_offset + 1,
            tl.where(active, rotated_plus_imag, 0.0),
            mask=in_output,
        )
        tl.store(
            cross_ptr + output_offset,
            tl.where(active, rotated_cross_real, 0.0),
            mask=in_output,
        )
        tl.store(
            cross_ptr + output_offset + 1,
            tl.where(active, rotated_cross_imag, 0.0),
            mask=in_output,
        )


def is_available():
    """Return whether the optional Triton CUDA implementation can be imported."""
    return _HAS_TRITON


def evaluate_taylorf2(
    coeff,
    coeff_log,
    coeff_log_sq,
    pi_mass,
    reference_phase,
    amplitude_factor,
    coa_phase,
    cos_inclination,
    cos_nodes,
    sin_nodes,
    first_bins,
    end_bins,
    delta_f,
    output_length,
):
    """Launch one kernel for a validated, non-differentiable CUDA batch.

    Coefficients have shape ``(16, batch)``. No launch or compilation error is
    caught: an explicitly selected path must not conceal a failed execution.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton CUDA support is unavailable")
    batch_size = pi_mass.numel()
    row_params = torch.stack(
        (
            pi_mass,
            reference_phase,
            amplitude_factor,
            coa_phase,
            cos_inclination,
            cos_nodes,
            sin_nodes,
        )
    )
    plus = torch.empty(
        (batch_size, output_length), dtype=torch.complex128, device=pi_mass.device
    )
    cross = torch.empty_like(plus)
    block = 128
    with torch.cuda.device(pi_mass.device):
        _taylorf2_kernel[(triton.cdiv(output_length, block), batch_size)](
            torch.view_as_real(plus),
            torch.view_as_real(cross),
            coeff.contiguous(),
            coeff_log.contiguous(),
            coeff_log_sq.contiguous(),
            row_params,
            first_bins,
            end_bins,
            batch_size,
            output_length,
            delta_f,
            2.0 * math.pi * (-1.0 / delta_f),
            math.pi / 4.0,
            BLOCK=block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return plus, cross
