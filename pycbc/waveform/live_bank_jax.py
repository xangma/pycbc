# Copyright (C) 2026 The PyCBC Collaboration
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

"""JAX-native live template generation and bank packing."""

import functools
import types
import jax
import jax.numpy as jnp

from pycbc.constants import MTSUN_SI, PI
from pycbc.types.array_jax import _ensure_x64
from pycbc.waveform.bank import LazyFrequencySeries, sigma_cached
from pycbc.waveform.taylorf2_jax import (
    _samples_core,
    _coefficients,
    fd_supported,
)
from pycbc.waveform.waveform import get_waveform_filter_length_in_time
import pycbc.pnutils as pnutils


@functools.partial(jax.jit, static_argnames=("flen",))
def _batch_generate_taylorf2_core(
    freqs_safe,
    pi_masses,
    coeffs,
    coeff_logs,
    m1s,
    m2s,
    etas,
    f_lows,
    f_finals,
    distances,
    flen,
):
    """Vectorized TaylorF2 generator over batch dimension via vmap."""
    def gen_one(pm, c, cl, m1, m2, eta, flow, ffin, dist):
        active = _samples_core(
            freqs_safe, pm, c, cl, 0.0, 0.0, m1, m2, dist, eta, jnp.complex64
        )
        mask = (freqs_safe >= flow) & (freqs_safe < ffin)
        return jnp.where(mask, active, 0.0)

    return jax.vmap(gen_one)(
        pi_masses,
        coeffs,
        coeff_logs,
        m1s,
        m2s,
        etas,
        f_lows,
        f_finals,
        distances,
    )


def generate_live_waveforms_jax(bank, rank, size):
    """Generate live template bank waveforms on the active JAX device.

    Templates sharing frequency resolution are batch-synthesized via vmap
    into device-resident 2D arrays and wrapped in LazyFrequencySeries to
    maintain 100% device residency without host copies.
    """
    _ensure_x64()

    if size > 1:
        indices = list(range(rank - 1, len(bank), size - 1))
    else:
        indices = list(range(len(bank)))

    if not indices:
        return []

    groups = {}
    for idx in indices:
        df = bank.freq_resolution_for_template(idx)
        groups.setdefault(df, []).append(idx)

    waveforms = [None] * len(indices)
    idx_to_pos = {idx: i for i, idx in enumerate(indices)}

    for df, g_indices in groups.items():
        flen = round(bank.sample_rate / (2 * df) + 1)
        params_list = [
            bank._waveform_parameters(idx, df)[0] for idx in g_indices
        ]

        all_taylorf2 = all(
            p.get("approximant") == "TaylorF2" and fd_supported(p)
            for p in params_list
        )

        if all_taylorf2:
            coeffs, coeff_logs, m1s, m2s, f_lows, f_finals, dists = (
                [], [], [], [], [], [], []
            )
            for idx, p in zip(g_indices, params_list):
                c, cl, _ = _coefficients(p)
                coeffs.append(c)
                coeff_logs.append(cl)
                m1s.append(float(p["mass1"]))
                m2s.append(float(p["mass2"]))
                f_lows.append(float(p["f_lower"]))
                f_fin = bank.end_frequency(idx)
                if f_fin is None or f_fin >= (flen * df):
                    f_fin = (flen - 1) * df
                f_finals.append(float(f_fin))
                dists.append(float(pnutils.megaparsecs_to_meters(
                    float(p.get("distance", 1.0))
                )))

            coeffs = jnp.stack(coeffs)
            coeff_logs = jnp.stack(coeff_logs)
            m1s_j = jnp.array(m1s)
            m2s_j = jnp.array(m2s)
            mts = m1s_j + m2s_j
            etas = m1s_j * m2s_j / (mts ** 2)
            pi_masses = PI * mts * MTSUN_SI
            f_lows_j = jnp.array(f_lows)
            f_finals_j = jnp.array(f_finals)
            dists_j = jnp.array(dists)

            freqs_full = jnp.arange(flen, dtype=jnp.float64) * df
            freqs_safe = jnp.where(freqs_full > 0, freqs_full, 1.0)

            w_tensor = _batch_generate_taylorf2_core(
                freqs_safe,
                pi_masses,
                coeffs,
                coeff_logs,
                m1s_j,
                m2s_j,
                etas,
                f_lows_j,
                f_finals_j,
                dists_j,
                flen,
            )

            for pos, (idx, p) in enumerate(zip(g_indices, params_list)):
                series = LazyFrequencySeries(w_tensor, pos, df)
                duration = get_waveform_filter_length_in_time(**p)
                bank.table[idx].template_duration = duration
                series.f_lower = f_lows[pos]
                series.min_f_lower = bank.min_f_lower
                series.end_idx = int(f_finals[pos] / df)
                series.end_frequency = f_finals[pos]
                series.params = bank.table[idx]
                series.chirp_length = duration
                series.length_in_time = duration
                series.approximant = bank.approximant(idx)
                series.sigmasq = types.MethodType(sigma_cached, series)
                series._sigmasq = {}
                series.id = bank.id_from_param((
                    series.params.mass1,
                    series.params.mass2,
                    series.params.spin1z,
                    series.params.spin2z,
                ))
                waveforms[idx_to_pos[idx]] = series
        else:
            for idx in g_indices:
                waveforms[idx_to_pos[idx]] = bank.get_template(idx, delta_f=df)

    return waveforms
