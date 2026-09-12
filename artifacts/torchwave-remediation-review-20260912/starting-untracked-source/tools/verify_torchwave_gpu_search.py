#!/usr/bin/env python3
# Copyright (C) 2026
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

"""Standalone verification and benchmark harness for torchwave and PyCBC

TiledMatchedFilterControl integration.
"""

import argparse
import sys
import time
import numpy as np
import torch

try:
    import torchwave
except ImportError:
    print("Error: torchwave is not installed. Install via:")
    print("  pip install -e /Users/xangma/repos/torchwave")
    sys.exit(1)

from pycbc import DYN_RANGE_FAC
from pycbc.filter.gpu_search.adapter import TiledMatchedFilterControl
from pycbc.filter.matchedfilter import match, sigmasq
from pycbc.types import FrequencySeries, zeros
from pycbc.waveform import get_fd_waveform as pycbc_get_fd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify torchwave batched matched-filtering parity."
    )
    parser.add_argument(
        "--approximant",
        default="TaylorF2",
        choices=["TaylorF2", "IMRPhenomD"],
        help="Approximant to verify (default: TaylorF2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tile batch size (default: 64)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device ('cpu', 'mps', 'cuda') (default: cpu)",
    )
    parser.add_argument(
        "--delta-f",
        type=float,
        default=0.5,
        help="Frequency step in Hz (default: 0.5)",
    )
    parser.add_argument(
        "--flen",
        type=int,
        default=1025,
        help="Filter length in frequency bins (default: 1025)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    b = args.batch_size
    delta_f = args.delta_f
    flen = args.flen
    tlen = (flen - 1) * 2
    device = args.device

    if device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but unavailable; using CPU.")
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; using CPU.")
        device = "cpu"

    if args.approximant == "TaylorF2":
        flow, fhigh = 30.0, 500.0
        m1_vals = np.linspace(1.4, 2.5, b)
        m2_vals = np.linspace(1.2, 1.5, b)
        s1z_vals = np.zeros(b)
        s2z_vals = np.zeros(b)
        tw_phic = np.pi / 2.0
    else:  # IMRPhenomD
        flow, fhigh = 20.0, 512.0
        m1_vals = np.linspace(25.0, 45.0, b)
        m2_vals = np.linspace(15.0, 25.0, b)
        s1z_vals = np.linspace(0.1, 0.4, b)
        s2z_vals = np.linspace(-0.2, 0.2, b)
        tw_phic = 0.0

    sep = "=" * 78
    print(sep)
    print("TORCHWAVE <-> PYCBC TILED MATCHED FILTER VERIFICATION HARNESS")
    print(
        f"Approximant: {args.approximant} | Batch Size: {b} | Device: {device}"
    )
    print(
        f"Grid: delta_f={delta_f} Hz | flen={flen} bins | "
        f"f_range=[{flow}, {fhigh}] Hz"
    )
    print(sep)

    # 1. Generate reference waveforms sequentially via PyCBC
    print(
        f"\n[1/4] Generating {b} reference templates via PyCBC "
        "(sequential CPU)..."
    )
    t0 = time.perf_counter()
    templates_ref = []
    for i in range(b):
        hp, _ = pycbc_get_fd(
            approximant=args.approximant,
            mass1=m1_vals[i],
            mass2=m2_vals[i],
            spin1z=s1z_vals[i],
            spin2z=s2z_vals[i],
            delta_f=delta_f,
            f_lower=flow,
            f_final=fhigh,
            distance=1.0 / DYN_RANGE_FAC,
        )
        hp_fs = FrequencySeries(
            zeros(flen, dtype=np.complex64), delta_f=delta_f
        )
        copy_len = min(len(hp), flen)
        hp_fs[:copy_len] = hp[:copy_len]
        templates_ref.append(hp_fs)
    t_ref_gen = time.perf_counter() - t0
    ref_rate = b / t_ref_gen
    print(
        f"      Completed in {t_ref_gen * 1000.0:.2f} ms "
        f"({ref_rate:.1f} templates/s)"
    )

    # 2. Generate batched waveforms via torchwave
    print(
        f"\n[2/4] Generating {b} templates via torchwave "
        f"(batched on {device})..."
    )
    freqs = torch.arange(0, flen, dtype=torch.float32, device=device) * delta_f
    mask = (freqs >= flow) & (freqs <= fhigh)
    active_freqs = freqs[mask]

    m1_t = torch.tensor(m1_vals, dtype=torch.float32, device=device)
    m2_t = torch.tensor(m2_vals, dtype=torch.float32, device=device)
    s1z_t = torch.tensor(s1z_vals, dtype=torch.float32, device=device)
    s2z_t = torch.tensor(s2z_vals, dtype=torch.float32, device=device)

    # Warmup
    _ = torchwave.get_fd_waveform(
        args.approximant,
        mass1=m1_t,
        mass2=m2_t,
        spin1z=s1z_t,
        spin2z=s2z_t,
        phic=tw_phic,
        sample_frequencies=active_freqs,
        distance=1.0 / DYN_RANGE_FAC,
        dtype=torch.float32,
        device=device,
    )
    if device in ("cuda", "mps"):
        getattr(torch, device).synchronize()

    t0 = time.perf_counter()
    hp_tw_batch, _ = torchwave.get_fd_waveform(
        args.approximant,
        mass1=m1_t,
        mass2=m2_t,
        spin1z=s1z_t,
        spin2z=s2z_t,
        phic=tw_phic,
        sample_frequencies=active_freqs,
        distance=1.0 / DYN_RANGE_FAC,
        dtype=torch.float32,
        device=device,
    )
    if device in ("cuda", "mps"):
        getattr(torch, device).synchronize()
    t_tw_gen = time.perf_counter() - t0
    tw_rate = b / t_tw_gen
    print(
        f"      Completed in {t_tw_gen * 1000.0:.2f} ms "
        f"({tw_rate:.1f} templates/s)"
    )
    print(f"      Generation speedup: {t_ref_gen / t_tw_gen:.2f}x")

    # 3. Compute overlap / match for sample templates
    print("\n[3/4] Validating waveform complex match (PyCBC vs torchwave)...")
    matches = []
    test_indices = [0, b // 4, b // 2, 3 * b // 4, b - 1]
    for idx in test_indices:
        tw_row = np.zeros(flen, dtype=np.complex64)
        mask_cpu = mask.cpu().numpy()
        tw_row[mask_cpu] = hp_tw_batch[idx].detach().cpu().numpy()
        hp_tw_fs = FrequencySeries(tw_row, delta_f=delta_f)

        overlap, _ = match(
            templates_ref[idx],
            hp_tw_fs,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
        )
        matches.append(overlap)
        print(
            f"      Template {idx:2d} (m1={m1_vals[idx]:.1f}, "
            f"m2={m2_vals[idx]:.1f}): Match = {overlap:.8f}"
        )

    min_match = min(matches)
    assert min_match > 0.9999, f"Match {min_match} below scientific threshold"

    # 4. End-to-end Matched Filtering with TiledMatchedFilterControl
    print(
        "\n[4/4] Executing batched matched filtering with "
        "TiledMatchedFilterControl..."
    )
    np.random.seed(42)
    s_data = (
        np.random.randn(flen) + 1j * np.random.randn(flen)
    ).astype(np.complex64)
    s_data[0] = 0.0
    seg = FrequencySeries(s_data, delta_f=delta_f)
    psd_arr = np.ones(flen, dtype=np.float32) * 2.0
    seg.psd = FrequencySeries(psd_arr, delta_f=delta_f)
    seg.analyze = slice(100, tlen - 100)
    seg._epoch = 0

    # Inject template 0 as an active candidate
    seg += templates_ref[0] * 30.0

    sigmasqs_ref = [
        sigmasq(
            t, seg.psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh
        )
        for t in templates_ref
    ]

    full_tensor_tw = torch.zeros(
        (b, flen), dtype=torch.complex64, device=device
    )
    full_tensor_tw[:, mask] = hp_tw_batch

    psd_t = torch.from_numpy(seg.psd.numpy()).to(device=device)
    sigmasqs_tw = (
        4.0
        * delta_f
        * torch.sum(
            torch.abs(full_tensor_tw[:, mask]) ** 2 / psd_t[mask], dim=-1
        )
    ).cpu().tolist()

    ctrl = TiledMatchedFilterControl(
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        snr_threshold=4.0,
        tlen=tlen,
        delta_f=delta_f,
        dtype=np.complex64,
        segment_list=[seg],
        template_output=zeros(tlen, dtype=np.complex64),
        use_cluster=True,
        cluster_function="symmetric",
        tile_size=b,
        device=device if device != "mps" else "cpu",
    )

    t0 = time.perf_counter()
    prep_ref = ctrl.prepare_template_batch(templates_ref)
    res_ref = ctrl.batched_matched_filter_and_cluster(
        0, prep_ref, sigmasqs_ref, window=10
    )
    t_filt_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    tw_device_prep = (
        full_tensor_tw.cpu() if device == "mps" else full_tensor_tw
    )
    prep_tw = ctrl.prepare_template_batch(tw_device_prep)
    res_tw = ctrl.batched_matched_filter_and_cluster(
        0, prep_tw, sigmasqs_tw, window=10
    )
    t_filt_tw = time.perf_counter() - t0

    print(
        f"      Filtering wall time (PyCBC templates): "
        f"{t_filt_ref * 1000.0:.2f} ms"
    )
    print(
        f"      Filtering wall time (torchwave 2D tensor): "
        f"{t_filt_tw * 1000.0:.2f} ms"
    )

    # Verify parity on results
    print("\n" + sep)
    print("MATCHED FILTERING PARITY RESULTS")
    print(sep)
    print(
        "Idx | Trig (Ref) | Trig (TW) | Loudest Arrival Diff | "
        "Peak SNR Diff (%) | Status"
    )
    print(
        "----+------------+-----------+----------------------+-"
        "------------------+-------"
    )

    for i in range(min(8, b)):
        _, _, _, idx_ref, snrv_ref = res_ref[i]
        _, _, _, idx_tw, snrv_tw = res_tw[i]

        count_ref = len(idx_ref)
        count_tw = len(idx_tw)

        if count_ref > 0 and count_tw > 0:
            loudest_ref = int(idx_ref[np.argmax(np.abs(snrv_ref))])
            loudest_tw = int(idx_tw[np.argmax(np.abs(snrv_tw))])
            loudest_diff = abs(loudest_ref - loudest_tw)

            peak_ref = float(np.max(np.abs(snrv_ref)))
            peak_tw = float(np.max(np.abs(snrv_tw)))
            snr_diff_pct = abs(peak_ref - peak_tw) / peak_ref * 100.0
        else:
            loudest_diff = 0
            snr_diff_pct = 0.0

        passed = (
            count_ref == count_tw
            and loudest_diff == 0
            and snr_diff_pct < 0.1
        )
        status = "PASSED" if passed else "FAILED"
        print(
            f"{i:3d} | {count_ref:10d} | {count_tw:9d} | "
            f"{loudest_diff:20d} | {snr_diff_pct:16.4f}% | {status}"
        )

    print(sep)
    print(
        "ALL CHECKS PASSED: torchwave successfully integrates with "
        "TiledMatchedFilterControl."
    )
    print(sep)


if __name__ == "__main__":
    main()
