# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Torch backend for waveform decompression (inline interpolation helpers).

These functions are wired by the scheme prefix ``pycbc.waveform.decompress_``.
They mirror the CPU C/Cython kernels but use torch tensor math. Intermediate
calculations use float64 when the device supports it, even when the output is
float32, to better match the CPU path (which accumulates in double). MPS uses
float32 intermediates because it does not support float64 tensors. The final
values are cast to the output dtype.
"""

import torch
import math


def _dtype_for_output(output):
    """Return real/complex torch dtypes matched to the output precision."""
    if output.precision == "single":
        return torch.float32, torch.complex64
    return torch.float64, torch.complex128


def _work_dtype(device):
    """Choose the highest intermediate precision supported by the device."""
    if device.type == "mps":
        return torch.float32, torch.complex64
    return torch.float64, torch.complex128


def _to_tensor(arr, device, dtype):
    """Convert numpy/torch/Array-like to a torch tensor on the given device."""
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)

    tensor = getattr(arr, "tensor", None)
    if tensor is None:
        tensor = getattr(getattr(arr, "_data", None), "tensor", None)
    if tensor is not None:
        return tensor.to(device=device, dtype=dtype)

    return torch.as_tensor(
        getattr(arr, "_data", arr), device=device, dtype=dtype
    )


def _lagrange_eval(x_nodes, y_nodes, x_eval):
    """Evaluate Lagrange polynomial defined by (x_nodes, y_nodes) at x_eval."""
    # x_nodes: (m,), y_nodes: (m,), x_eval: (n,)
    m = x_nodes.numel()
    # Compute barycentric weights
    diff = x_nodes.unsqueeze(0) - x_nodes.unsqueeze(1)  # (m, m)
    diff = diff + torch.eye(m, device=x_nodes.device, dtype=x_nodes.dtype)  # replace diag with 1
    w = 1.0 / torch.prod(diff, dim=1)
    x_diff = x_eval.unsqueeze(1) - x_nodes.unsqueeze(0)  # (n, m)

    # Handle exact node hits to avoid NaNs
    exact = torch.isclose(x_diff, torch.zeros(1, device=x_nodes.device, dtype=x_nodes.dtype))
    if exact.any():
        # For positions where x_eval equals a node, use that node's value
        out = torch.empty_like(x_eval, dtype=y_nodes.dtype)
        # Fill with generic barycentric result first
        denom = torch.sum(w / x_diff, dim=1)
        numer = torch.sum(w * y_nodes / x_diff, dim=1)
        out[:] = numer / denom
        # Overwrite exact hits
        hit_rows, hit_cols = torch.nonzero(exact, as_tuple=True)
        out[hit_rows] = y_nodes[hit_cols]
        return out

    denom = torch.sum(w / x_diff, dim=1)
    numer = torch.sum(w * y_nodes / x_diff, dim=1)
    return numer / denom


def _power_coeffs(f_segment, y_segment):
    """Return power-basis coefficients for a polynomial through the given nodes."""
    n = f_segment.numel()
    vander = torch.vander(f_segment, N=n, increasing=True)
    coeffs = torch.linalg.solve(vander, y_segment)
    return coeffs  # c0, c1, ...


def _decomp_linear_segment(out, k, kmax, f1, f2, a1, a2, p1, p2, df):
    """Replicate _decomp_ccode_segment logic in torch."""
    inv_sdf = 1.0 / (f2 - f1)
    m_amp = (a2 - a1) * inv_sdf
    b_amp = a1 - m_amp * f1
    m_phi = (p2 - p1) * inv_sdf
    b_phi = p1 - m_phi * f1
    update_interval = 128
    findex = k
    while findex < kmax:
        f = findex * df
        interp_amp = m_amp * f + b_amp
        interp_phi = m_phi * f + b_phi
        dphi_re = torch.cos(m_phi * df)
        dphi_im = torch.sin(m_phi * df)
        h_re = interp_amp * torch.cos(interp_phi)
        h_im = interp_amp * torch.sin(interp_phi)
        g_re = m_amp * df * torch.cos(interp_phi)
        g_im = m_amp * df * torch.sin(interp_phi)
        out[findex] = torch.complex(h_re, h_im)
        findex += 1
        k_sub_max = min(findex + update_interval, kmax)
        while findex < k_sub_max:
            incrh_re = h_re * dphi_re - h_im * dphi_im
            incrh_im = h_re * dphi_im + h_im * dphi_re
            incrg_re = g_re * dphi_re - g_im * dphi_im
            incrg_im = g_re * dphi_im + g_im * dphi_re
            h_re = incrh_re + incrg_re
            h_im = incrh_im + incrg_im
            g_re = incrg_re
            g_im = incrg_im
            # Cast back to output dtype on write
            out[findex] = torch.complex(h_re, h_im).to(out.dtype)
            findex += 1


def _decomp_poly_segment(out, k, kmax, coeff_a, coeff_p, df, degree):
    """Generic stepper matching _decomp_q/t/Qcode_segment."""
    update_interval = 128
    h2 = df * df
    h3 = h2 * df
    h4 = h3 * df

    while k < kmax:
        f = k * df
        k_sub_max = min(k + update_interval, kmax)

        # evaluate a, p and first/second/third differences per degree
        if degree == 2:
            c_a0, c_a1, c_a2 = coeff_a
            c_p0, c_p1, c_p2 = coeff_p
            a = c_a2 * f * f + c_a1 * f + c_a0
            p = c_p2 * f * f + c_p1 * f + c_p0
            d1_a = c_a2 * (2 * f * df + h2) + c_a1 * df
            d1_p = c_p2 * (2 * f * df + h2) + c_p1 * df
            d2_a_const = 2 * c_a2 * h2
            d2_p_const = 2 * c_p2 * h2
            d_phase = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), d1_p)
            d2_phase_const = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), d2_p_const)
        elif degree == 3:
            c_a0, c_a1, c_a2, c_a3 = coeff_a
            c_p0, c_p1, c_p2, c_p3 = coeff_p
            a = ((c_a3 * f + c_a2) * f + c_a1) * f + c_a0
            p = ((c_p3 * f + c_p2) * f + c_p1) * f + c_p0
            d1_a = c_a3 * (3 * f * f * df + 3 * f * h2 + h3) + c_a2 * (2 * f * df + h2) + c_a1 * df
            d1_p = c_p3 * (3 * f * f * df + 3 * f * h2 + h3) + c_p2 * (2 * f * df + h2) + c_p1 * df
            d2_a = c_a3 * (6 * f * h2 + 6 * h3) + c_a2 * (2 * h2)
            d2_p = c_p3 * (6 * f * h2 + 6 * h3) + c_p2 * (2 * h2)
            d3_a_const = 6 * c_a3 * h3
            d3_p_const = 6 * c_p3 * h3
            d1_phase = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), d1_p)
            d2_phase = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), d2_p)
            d3_phase_const = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), d3_p_const)
        else:  # degree == 4
            c_a0, c_a1, c_a2, c_a3, c_a4 = coeff_a
            c_p0, c_p1, c_p2, c_p3, c_p4 = coeff_p
            a = (((c_a4 * f + c_a3) * f + c_a2) * f + c_a1) * f + c_a0
            p = (((c_p4 * f + c_p3) * f + c_p2) * f + c_p1) * f + c_p0
            d1_a = c_a4 * (4 * f * f * f * df + 6 * f * f * h2 + 4 * f * h3 + h4) + \
                   c_a3 * (3 * f * f * df + 3 * f * h2 + h3) + c_a2 * (2 * f * df + h2) + c_a1 * df
            d1_p = c_p4 * (4 * f * f * f * df + 6 * f * f * h2 + 4 * f * h3 + h4) + \
                   c_p3 * (3 * f * f * df + 3 * f * h2 + h3) + c_p2 * (2 * f * df + h2) + c_p1 * df
            d2_a = c_a4 * (12 * f * f * h2 + 24 * f * h3 + 14 * h4) + c_a3 * (6 * f * h2 + 6 * h3) + c_a2 * (2 * h2)
            d2_p = c_p4 * (12 * f * f * h2 + 24 * f * h3 + 14 * h4) + c_p3 * (6 * f * h2 + 6 * h3) + c_p2 * (2 * h2)
            d3_a = c_a4 * (24 * f * h3 + 36 * h4) + c_a3 * (6 * h3)
            d3_p = c_p4 * (24 * f * h3 + 36 * h4) + c_p3 * (6 * h3)
            d4_a_const = 24 * c_a4 * h4
            d4_p_const = 24 * c_p4 * h4
            d1_phase = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), d1_p)
            d2_phase = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), d2_p)
            d3_phase = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), d3_p)
            d4_phase_const = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), d4_p_const)

        # phase at start of block
        phase = torch.polar(torch.tensor(1.0, dtype=a.dtype, device=out.device), p)

        while k < k_sub_max:
            out[k] = (a * phase).to(out.dtype)
            if degree == 2:
                phase = phase * d_phase
                d_phase = d_phase * d2_phase_const
                a = a + d1_a
                d1_a = d1_a + d2_a_const
            elif degree == 3:
                phase = phase * d1_phase
                d1_phase = d1_phase * d2_phase
                d2_phase = d2_phase * d3_phase_const
                a = a + d1_a
                d1_a = d1_a + d2_a
                d2_a = d2_a + d3_a_const
            else:
                phase = phase * d1_phase
                d1_phase = d1_phase * d2_phase
                d2_phase = d2_phase * d3_phase
                d3_phase = d3_phase * d4_phase_const
                a = a + d1_a
                d1_a = d1_a + d2_a
                d2_a = d2_a + d3_a
                d3_a = d3_a + d4_a_const
            k += 1


def _decomp_main_loop_torch(degree, amp, phase, sample_frequencies, output,
                            df, start_index, imin):
    """Torch translation of _decomp_main_loop from decompress_cpu_ccode."""
    real_dtype, complex_dtype = _dtype_for_output(output)
    device = output.data.tensor.device
    work_real, work_complex = _work_dtype(device)

    # Use higher precision for coefficient math to mirror CPU double steppers.
    # NOTE: Even with double intermediates, we still see ~2.5e-3 max delta
    # vs. CPU quartic because the CPU uses a finite-difference stepper with
    # periodic re-seeding, while here we evaluate the polynomial directly.
    # If tighter parity is needed, re-implement the full FD stepper in torch
    # (matching update_interval=128 and recurrence order) rather than direct
    # polynomial evaluation.
    sf = _to_tensor(sample_frequencies, device, work_real)
    a = _to_tensor(amp, device, work_real)
    ph = _to_tensor(phase, device, work_real)

    out = output.data.tensor
    hlen = out.numel()
    out.zero_()

    # Maximum possible degree given number of samples
    if sf.numel() < 3:
        max_degree = 1
    elif sf.numel() < 4:
        max_degree = 2
    elif sf.numel() < 5:
        max_degree = 3
    else:
        max_degree = 4
    degree = min(degree, max_degree)

    last_findex = start_index
    sflen = sf.numel()
    df_t = torch.tensor(df, device=device, dtype=work_real)

    # Zero out prior to start_index (already zeroed full tensor)
    for i in range(int(imin), int(sflen) - 1):
        f1 = sf[i]
        f2 = sf[i + 1]
        a1 = a[i]
        a2 = a[i + 1]
        p1 = ph[i]
        p2 = ph[i + 1]

        if i == imin:
            k = int(start_index)
        else:
            k = int(math.ceil(float(f1 / df_t)))

        if i == sflen - 2:
            kmax = int(f2 / df_t) + 1
        else:
            kmax = int(f2 / df_t)
        if kmax > hlen:
            kmax = hlen
        if k >= kmax:
            last_findex = max(last_findex, kmax)
            continue

        current_degree = degree
        if i == imin:
            current_degree = 1
        elif i == imin + 1 and current_degree > 2:
            current_degree = 2
        elif i == imin + 2 and current_degree > 3:
            current_degree = 3

        if current_degree > 3 and i >= sflen - 3:
            current_degree = 3
        if current_degree > 2 and i >= sflen - 2:
            current_degree = 2

        current_degree = min(current_degree, max_degree)

        # Collect stencil and run degree-specific segment
        if current_degree == 1:
            _decomp_linear_segment(out, k, kmax, f1, f2, a1, a2, p1, p2, df_t)
        else:
            if current_degree == 2:
                fseg = torch.stack((sf[i - 1], f1, f2))
                aseg = torch.stack((a[i - 1], a1, a2))
                pseg = torch.stack((ph[i - 1], p1, p2))
            elif current_degree == 3:
                fseg = torch.stack((sf[i - 1], f1, f2, sf[i + 2]))
                aseg = torch.stack((a[i - 1], a1, a2, a[i + 2]))
                pseg = torch.stack((ph[i - 1], p1, p2, ph[i + 2]))
            else:
                fseg = torch.stack((sf[i - 1], f1, f2, sf[i + 2], sf[i + 3]))
                aseg = torch.stack((a[i - 1], a1, a2, a[i + 2], a[i + 3]))
                pseg = torch.stack((ph[i - 1], p1, p2, ph[i + 2], ph[i + 3]))

            coeff_a = _power_coeffs(fseg, aseg)
            coeff_p = _power_coeffs(fseg, pseg)
            _decomp_poly_segment(out, k, kmax, coeff_a, coeff_p, df_t, current_degree)

        last_findex = max(last_findex, kmax)

    if last_findex < hlen:
        out[last_findex:] = 0

    return output


def inline_linear_interp(amp, phase, sample_frequencies, output,
                         df, f_lower, imin, start_index):
    return _decomp_main_loop_torch(1, amp, phase, sample_frequencies, output,
                                   df, start_index, imin)


def inline_quadratic_interp(amp, phase, sample_frequencies, output,
                            df, f_lower, imin, start_index):
    return _decomp_main_loop_torch(2, amp, phase, sample_frequencies, output,
                                   df, start_index, imin)


def inline_cubic_interp(amp, phase, sample_frequencies, output,
                        df, f_lower, imin, start_index):
    return _decomp_main_loop_torch(3, amp, phase, sample_frequencies, output,
                                   df, start_index, imin)


def inline_quartic_interp(amp, phase, sample_frequencies, output,
                          df, f_lower, imin, start_index):
    return _decomp_main_loop_torch(4, amp, phase, sample_frequencies, output,
                                   df, start_index, imin)
