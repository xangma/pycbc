#!/usr/bin/env python
"""Generate publication-grade performance plots for PyCBC Torch docs."""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Apply clean modern styling
style = ('seaborn-v0_8-whitegrid'
         if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.style.use(style)
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 2.2,
    'lines.markersize': 7,
})

# Output directories
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_docs_img_dir = os.path.join(repo_root, "docs", "images")

parser = argparse.ArgumentParser(
    description="Generate PyCBC performance plots."
)
parser.add_argument(
    "--artifacts-dir",
    required=True,
    help="Directory containing the complete benchmark JSON inputs",
)
parser.add_argument("--output-dir", default=default_docs_img_dir)
cli_args = parser.parse_args()

artifacts_dir = cli_args.artifacts_dir
docs_img_dir = cli_args.output_dir

os.makedirs(docs_img_dir, exist_ok=True)

# Colors
c_std = '#4A5568'    # Slate Gray
c_tcpu = '#3182CE'   # Torch Blue
c_tcuda = '#38A169'  # Emerald Green
c_dual = '#805AD5'   # Purple

# ==============================================================================
# DATA INGESTION: complete measured artifacts are mandatory
# ==============================================================================
def _find_artifact(description, *names):
    for name in names:
        path = os.path.join(artifacts_dir, name)
        if os.path.isfile(path):
            return path
    parser.error(
        f"missing {description}; expected one of "
        + ", ".join(os.path.join(artifacts_dir, name) for name in names)
    )


def _median(obj):
    if not isinstance(obj, dict) or "median" not in obj:
        raise ValueError("benchmark route has no median measurement")
    return obj["median"]


def _to_ms(obj):
    return _median(obj) * 1000.0


def _to_tput(batch, obj):
    return batch / _median(obj)


def _get_route(result, *names):
    for name in names:
        if name in result:
            return result[name]
    raise ValueError(f"benchmark result is missing routes {names!r}")


suite_json = _find_artifact(
    "comprehensive benchmark artifact",
    "extended_full_suite.json",
    "comprehensive_benchmark_results.json",
    "len_extended_full_suite.json",
    "live_batch_latest.json",
)
try:
    with open(suite_json, "r", encoding="utf-8") as artifact_file:
        suite_data = json.load(artifact_file)
    results = suite_data["results"]
    metadata = suite_data["metadata"]

    required_batches = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    live_results = results["synthetic_live_batch"]
    fft_results = results["correlate_ifft"]
    if any(f"batch_{batch}" not in live_results for batch in required_batches):
        raise ValueError("synthetic_live_batch does not contain every batch")
    if any(f"batch_{batch}" not in fft_results for batch in required_batches):
        raise ValueError("correlate_ifft does not contain every batch")

    batches = np.array(required_batches)
    std_lat = np.array([
        _to_ms(_get_route(live_results[f"batch_{batch}"],
                          "original_1t", "original_standard"))
        for batch in batches
    ])
    std_16t_lat = np.array([
        _to_ms(_get_route(live_results[f"batch_{batch}"],
                          "original_16t", "original_threaded_16t",
                          "branch_standard"))
        for batch in batches
    ])
    tcpu_lat = np.array([
        _to_ms(_get_route(live_results[f"batch_{batch}"],
                          "torch_cpu_16t", "torch_cpu"))
        for batch in batches
    ])
    tcuda_lat = np.array([
        _to_ms(_get_route(live_results[f"batch_{batch}"], "torch_cuda"))
        for batch in batches
    ])

    fft_batches = np.array(required_batches)
    fft_std_1t = np.array([
        _to_tput(batch, _get_route(fft_results[f"batch_{batch}"],
                                   "original_1t", "original_standard"))
        for batch in fft_batches
    ])
    fft_std_16t = np.array([
        _to_tput(batch, _get_route(fft_results[f"batch_{batch}"],
                                   "original_16t",
                                   "original_threaded_16t"))
        for batch in fft_batches
    ])
    fft_tcpu_16t = np.array([
        _to_tput(batch, _get_route(fft_results[f"batch_{batch}"],
                                   "torch_cpu_16t"))
        for batch in fft_batches
    ])
    fft_cuda = np.array([
        _to_tput(batch, _get_route(fft_results[f"batch_{batch}"],
                                   "torch_cuda"))
        for batch in fft_batches
    ])

    detector_points = [10, 100, 1000, 10000]
    detector_results = results["detector_network"]
    if any(f"points_{point}" not in detector_results
           for point in detector_points):
        raise ValueError("detector_network does not contain every grid size")
    n_sky = np.array(detector_points)
    net_seq_ms = np.array([
        _to_ms(_get_route(detector_results[f"points_{point}"],
                          "sequential", "sequential_single"))
        for point in n_sky
    ])
    net_vec_ms = np.array([
        _to_ms(_get_route(detector_results[f"points_{point}"],
                          "vectorized", "vectorized_network"))
        for point in n_sky
    ])

    relbin_samples = [10, 100, 1000, 10000]
    relbin_results = results["relbin_likelihood"]
    if any(f"samples_{sample}" not in relbin_results
           for sample in relbin_samples):
        raise ValueError("relbin_likelihood does not contain every batch")
    b_relbin = np.array(relbin_samples)
    relbin_seq_ms = np.array([
        _to_ms(_get_route(relbin_results[f"samples_{sample}"],
                          "sequential", "sequential_single"))
        for sample in b_relbin
    ])
    relbin_bat_ms = np.array([
        _to_ms(_get_route(relbin_results[f"samples_{sample}"],
                          "batched", "batched_summary"))
        for sample in b_relbin
    ])
    cuda_devices = metadata["cuda_devices"]
    benchmark_host = metadata["hostname"]
    cuda_name = cuda_devices[0]["name"]
    if not all(isinstance(value, str) and value.strip()
               for value in (benchmark_host, cuda_name)):
        raise ValueError("host and CUDA device metadata must be non-empty")
except (IndexError, KeyError, TypeError, ValueError,
        json.JSONDecodeError) as exc:
    parser.error(f"invalid comprehensive benchmark artifact {suite_json}: {exc}")

cuda_label = f"Torch CUDA ({cuda_name})"

std_tput = (batches / std_lat) * 1000.0
std_per_wf = std_lat / batches
std_16t_tput = (batches / std_16t_lat) * 1000.0
tcpu_tput = (batches / tcpu_lat) * 1000.0
tcpu_per_wf = tcpu_lat / batches
tcpu_speedup = std_lat / tcpu_lat
tcuda_tput = (batches / tcuda_lat) * 1000.0
tcuda_per_wf = tcuda_lat / batches
tcuda_speedup = std_lat / tcuda_lat

print(f"Loaded complete benchmark suite from {suite_json}")


def save_fig(fig, filename):
    output_path = os.path.join(docs_img_dir, filename)
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")


# ==============================================================================
# FIGURE 1: Matched Filter Search Throughput & Scaling (2-Panel)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

# Panel 1: Throughput
ax1.plot(
    batches, tcuda_tput, 'o-', color=c_tcuda, linewidth=2.2,
    label=cuda_label, zorder=4
)
ax1.plot(
    batches, tcpu_tput, 's-', color=c_tcpu, linewidth=1.8,
    label='Torch CPU (16T MKL/SIMD)', zorder=3
)
ax1.plot(
    batches, std_16t_tput, 'v-.', color='#718096', linewidth=1.5,
    label='Standard CPU (16T FFTW)', zorder=2
)
ax1.plot(
    batches, std_tput, 'd--', color=c_std, linewidth=1.5,
    label='Standard CPU (1T Reference)', zorder=1
)

ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_ylim(40, 80000)
ax1.set_xticks(batches)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel('Batch Size ($B$)')
ax1.set_ylabel('Search Throughput (Waveforms / Second)')
ax1.set_title(
    '(A) Matched Filter Search Throughput ($N=131,072$)\n'
    f'(Measured on {benchmark_host}; {cuda_name})'
)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

max_cuda_idx = np.argmax(tcuda_tput)
ax1.annotate(
    f'Peak: {tcuda_tput[max_cuda_idx]:,.0f} wf/s\n'
    f'({tcuda_speedup[max_cuda_idx]:.1f}x vs 1T CPU)',
    xy=(batches[max_cuda_idx], tcuda_tput[max_cuda_idx]),
    xytext=(128, 16000),
    arrowprops=dict(arrowstyle='->', color=c_tcuda, lw=1.5),
    bbox=dict(
        boxstyle='round,pad=0.4', facecolor='#E6FFFA',
        edgecolor=c_tcuda, alpha=0.9
    ),
    fontsize=9.5, fontweight='bold', color='#234E52'
)

# Panel 2: Speedup relative to Standard CPU Baseline
ax2.axhline(
    1.0, color=c_std, linestyle='--', linewidth=1.5,
    label='Standard CPU 1T Baseline (1.0x)'
)
ax2.plot(
    batches, tcuda_speedup, 'o-', color=c_tcuda, linewidth=2.2,
    label='Torch CUDA vs Std CPU 1T', zorder=4
)
ax2.plot(
    batches, std_16t_lat / tcuda_lat, '*-', color=c_dual, linewidth=1.8,
    label='Torch CUDA vs Std CPU 16T', zorder=4
)
ax2.plot(
    batches, tcpu_speedup, 's-', color=c_tcpu, linewidth=1.8,
    label='Torch CPU 16T vs Std CPU 1T', zorder=3
)
ax2.plot(
    batches, std_16t_lat / tcpu_lat, '^:', color='#319795', linewidth=1.6,
    label='Torch CPU 16T vs Std CPU 16T', zorder=3
)
ax2.plot(
    batches, std_16t_tput / std_tput, 'v-.', color='#718096', linewidth=1.5,
    label='Std CPU 16T vs Std CPU 1T', zorder=2
)

ax2.set_xscale('log', base=2)
ax2.set_xticks(batches)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.set_xlabel('Batch Size ($B$)')
ax2.set_ylabel('Speedup Multiplier')
ax2.set_title(
    '(B) Multi-Tier Acceleration Multipliers vs Batch Size\n'
    '(Relative to 1T Baseline and 16T Parallel Baseline)'
)
ax2.legend(
    frameon=True, facecolor='white', framealpha=0.9, loc='upper left',
    fontsize=8.5
)

max_sp_idx = np.argmax(tcuda_speedup)
ax2.annotate(
    f'Peak: {tcuda_speedup[max_sp_idx]:.1f}x Speedup\n'
    f'({tcuda_tput[max_sp_idx]:,.0f} wf/s)',
    xy=(batches[max_sp_idx], tcuda_speedup[max_sp_idx]),
    xytext=(64, 38),
    arrowprops=dict(arrowstyle='->', color=c_tcuda, lw=1.5),
    bbox=dict(
        boxstyle='round,pad=0.4', facecolor='#E6FFFA',
        edgecolor=c_tcuda, alpha=0.9
    ),
    fontsize=9.5, fontweight='bold', color='#234E52'
)

plt.tight_layout()
save_fig(fig, 'torch_live_batch_scaling.png')
plt.close()

# ==============================================================================
# FIGURE 2: Latency Per Block & Per-Waveform Cost
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: Latency per Block
ax1.plot(batches, std_lat, 'd--', color=c_std, label='Standard CPU (1T)')
ax1.plot(
    batches, std_16t_lat, 'v-.', color='#718096', label='Standard CPU (16T)'
)
ax1.plot(batches, tcpu_lat, 's-', color=c_tcpu, label='Torch CPU (16T)')
ax1.plot(
    batches, tcuda_lat, 'o-', color=c_tcuda, label=cuda_label
)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_ylim(0.2, 5000.0)
ax1.set_xticks(batches)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel('Batch Size ($B$)')
ax1.set_ylabel('Total Latency per Block (ms)')
ax1.set_title('Block Processing Latency (Lower is Better)')
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# Panel 2: Amortized Cost per Template Waveform
ax2.plot(batches, std_per_wf, 'd--', color=c_std, label='Standard CPU (1T)')
ax2.plot(
    batches, std_16t_lat / batches, 'v-.', color='#718096',
    label='Standard CPU (16T)'
)
ax2.plot(batches, tcpu_per_wf, 's-', color=c_tcpu, label='Torch CPU (16T)')
ax2.plot(
    batches, tcuda_per_wf, 'o-', color=c_tcuda, label=cuda_label
)
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_ylim(0.01, 15.0)
ax2.set_xticks(batches)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.set_xlabel('Batch Size ($B$)')
ax2.set_ylabel('Amortized Latency per Template (ms)')
ax2.set_title('Per-Waveform Amortized Cost (Lower is Better)')
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

pwf_idx = len(batches) - 1
ax2.annotate(
    f'{tcuda_per_wf[pwf_idx]*1000:.1f} µs/template\n'
    f'({tcuda_tput[pwf_idx]:,.0f} wf/s)',
    xy=(batches[pwf_idx], tcuda_per_wf[pwf_idx]),
    xytext=(4, 0.03),
    arrowprops=dict(arrowstyle='->', color=c_tcuda, lw=1.5),
    bbox=dict(
        boxstyle='round,pad=0.4', facecolor='#E6FFFA',
        edgecolor=c_tcuda, alpha=0.9
    ),
    fontsize=9.5, fontweight='bold', color='#234E52'
)

plt.tight_layout()
save_fig(fig, 'torch_latency_breakdown.png')
plt.close()
# ==============================================================================
# FIGURE 3: Comprehensive 4-Panel Executive Performance Dashboard
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Top-Left: Search Throughput
axs[0, 0].plot(
    batches, tcuda_tput, 'o-', color=c_tcuda, linewidth=2.0,
    label=cuda_label
)
axs[0, 0].plot(
    batches, tcpu_tput, 's-', color=c_tcpu, linewidth=1.6,
    label='Torch CPU (16T)'
)
axs[0, 0].plot(
    batches, std_16t_tput, 'v-.', color='#718096', linewidth=1.4,
    label='Standard CPU (16T)'
)
axs[0, 0].plot(
    batches, std_tput, 'd--', color=c_std, linewidth=1.4,
    label='Standard CPU (1T)'
)
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_yscale('log')
axs[0, 0].set_ylim(40, 80000)
axs[0, 0].set_xticks(batches)
axs[0, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axs[0, 0].set_xlabel('Batch Size ($B$)')
axs[0, 0].set_ylabel('Throughput (wf/s)')
axs[0, 0].set_title('(A) Matched Filtering Search Throughput')
axs[0, 0].legend(
    frameon=True, facecolor='white', framealpha=0.9, loc='upper left'
)
axs[0, 0].grid(True, which='both', linestyle='--', alpha=0.5)

# Top-Right: Speedup Factor
axs[0, 1].axhline(
    1.0, color=c_std, linestyle='--', linewidth=1.5,
    label='Standard CPU 1T Baseline'
)
axs[0, 1].plot(
    batches, tcuda_speedup, 'o-', color=c_tcuda, linewidth=2.0,
    label='Torch CUDA Speedup vs 1T'
)
axs[0, 1].plot(
    batches, tcpu_speedup, 's-', color=c_tcpu, linewidth=1.6,
    label='Torch CPU 16T Speedup vs 1T'
)
axs[0, 1].plot(
    batches, std_16t_tput / std_tput, 'v-.', color='#718096', linewidth=1.4,
    label='Standard CPU 16T vs 1T'
)
axs[0, 1].set_xscale('log', base=2)
axs[0, 1].set_xticks(batches)
axs[0, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axs[0, 1].set_xlabel('Batch Size ($B$)')
axs[0, 1].set_ylabel('Speedup Multiplier vs Standard CPU')
axs[0, 1].set_title('(B) Search Acceleration Multiplier vs Batch Size')
max_dash_sp = np.argmax(tcuda_speedup)
axs[0, 1].set_ylim(0, 65)
axs[0, 1].legend(
    frameon=True, facecolor='white', framealpha=0.9, loc='upper left'
)
axs[0, 1].grid(True, which='both', linestyle='--', alpha=0.5)

axs[0, 1].annotate(
    f'Peak: {tcuda_speedup[max_dash_sp]:.1f}x\n'
    f'({tcuda_tput[max_dash_sp]:,.0f} wf/s)',
    xy=(batches[max_dash_sp], tcuda_speedup[max_dash_sp]),
    xytext=(128, 12),
    arrowprops=dict(arrowstyle='->', color=c_tcuda, lw=1.5),
    bbox=dict(
        boxstyle='round,pad=0.3', facecolor='#E6FFFA',
        edgecolor=c_tcuda, alpha=0.9
    ),
    fontsize=9, fontweight='bold', color='#234E52'
)

# Bottom-Left: Per-Waveform Cost
axs[1, 0].plot(
    batches, std_per_wf, 'd--', color=c_std, label='Standard CPU (1T)'
)
axs[1, 0].plot(
    batches, std_16t_lat / batches, 'v-.', color='#718096',
    label='Standard CPU (16T)'
)
axs[1, 0].plot(
    batches, tcpu_per_wf, 's-', color=c_tcpu, label='Torch CPU (16T)'
)
axs[1, 0].plot(
    batches, tcuda_per_wf, 'o-', color=c_tcuda, label=cuda_label
)
axs[1, 0].set_xscale('log', base=2)
axs[1, 0].set_yscale('log')
axs[1, 0].set_ylim(0.01, 15.0)
axs[1, 0].set_xticks(batches)
axs[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axs[1, 0].set_xlabel('Batch Size ($B$)')
axs[1, 0].set_ylabel('Amortized Latency per Template (ms)')
axs[1, 0].set_title('(C) Per-Waveform Amortized Processing Cost')
axs[1, 0].legend(
    frameon=True, facecolor='white', framealpha=0.9, loc='upper right'
)
axs[1, 0].grid(True, which='both', linestyle='--', alpha=0.5)

# Bottom-Right: Pure FFT & Correlator Throughput
axs[1, 1].plot(
    fft_batches, fft_cuda, 'o-', color=c_tcuda, linewidth=2.0,
    label=cuda_label
)
axs[1, 1].plot(
    fft_batches, fft_tcpu_16t, 's-', color=c_tcpu, linewidth=1.6,
    label='Torch CPU (16T MKL)'
)
axs[1, 1].plot(
    fft_batches, fft_std_16t, 'v-.', color='#718096', linewidth=1.4,
    label='Standard CPU (16T FFTW)'
)
axs[1, 1].plot(
    fft_batches, fft_std_1t, 'd--', color=c_std, linewidth=1.4,
    label='Standard CPU (1T FFTW)'
)
axs[1, 1].set_xscale('log', base=2)
axs[1, 1].set_yscale('log')
axs[1, 1].set_ylim(200, 160000.0)
axs[1, 1].set_xticks(fft_batches)
axs[1, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axs[1, 1].set_xlabel('Batch Size ($B$)')
axs[1, 1].set_ylabel('Transforms / Second')
axs[1, 1].set_title('(D) Pure Correlator + IFFT Throughput ($N=131,072$)')
axs[1, 1].legend(
    frameon=True, facecolor='white', framealpha=0.9, loc='lower left'
)
axs[1, 1].grid(True, which='both', linestyle='--', alpha=0.5)

max_fft_idx = np.argmax(fft_cuda)
axs[1, 1].annotate(
    f'Peak: {fft_cuda[max_fft_idx]:,.0f} xf/s\n'
    f'({fft_cuda[max_fft_idx]/fft_std_1t[max_fft_idx]:.1f}x vs 1T CPU)',
    xy=(fft_batches[max_fft_idx], fft_cuda[max_fft_idx]),
    xytext=(16, 45000),
    arrowprops=dict(arrowstyle='->', color=c_tcuda, lw=1.5),
    bbox=dict(
        boxstyle='round,pad=0.3', facecolor='#E6FFFA',
        edgecolor=c_tcuda, alpha=0.9
    ),
    fontsize=9, fontweight='bold', color='#234E52'
)

plt.suptitle(
    'PyCBC PyTorch Acceleration Suite Performance Dashboard '
    f'({benchmark_host}; {cuda_name})',
    fontsize=14, y=0.995, fontweight='bold'
)
plt.tight_layout()
save_fig(fig, 'torch_performance_dashboard.png')
plt.close()
# ==============================================================================
# FIGURE 4: Inference & Multi-Detector Response Acceleration
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

# Panel 1: Multi-Detector NetworkGeometry (H1, L1, V1)
net_speedup = net_seq_ms / net_vec_ms

ax1.plot(
    n_sky, net_seq_ms, 'd--', color=c_std, linewidth=1.8,
    label='Sequential Detector Calls (1T CPU)'
)
ax1.plot(
    n_sky, net_vec_ms, 'o-', color=c_tcpu, linewidth=2.2,
    label='Vectorized NetworkGeometry (Torch/NumPy)'
)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(0.005, 10000.0)
ax1.set_xticks(n_sky)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel('Sky Grid Points ($N$)')
ax1.set_ylabel('Evaluation Latency (ms, log scale)')
ax1.set_title(
    '(A) Multi-Detector Antenna Patterns & Delays (H1-L1-V1)\n'
    'Sequential vs Vectorized Tensor Contractions'
)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

ax1.annotate(
    f'Peak: {net_speedup[-1]:.0f}x Speedup\n'
    f'({net_vec_ms[-1]:.2f} ms for {n_sky[-1]:,} sky points)',
    xy=(n_sky[-1], net_vec_ms[-1]),
    xytext=(300, 0.03),
    arrowprops=dict(arrowstyle='->', color=c_tcpu, lw=1.5),
    bbox=dict(
        boxstyle='round,pad=0.4', facecolor='#EBF8FF',
        edgecolor=c_tcpu, alpha=0.9
    ),
    fontsize=9.5, fontweight='bold', color='#2B6CB0'
)

# Panel 2: Batched Relative Binning Summary Evaluation (relbin_torch)
relbin_speedup = relbin_seq_ms / relbin_bat_ms

ax2.plot(
    b_relbin, relbin_seq_ms, 'd--', color=c_std, linewidth=1.8,
    label='Sequential Likelihood Parts (1T CPU)'
)
ax2.plot(
    b_relbin, relbin_bat_ms, 's-', color=c_tcuda, linewidth=2.2,
    label='Batched Summary Products (relbin_torch)'
)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim(0.005, 10000.0)
ax2.set_xticks(b_relbin)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.set_xlabel('Parameter Samples ($B$)')
ax2.set_ylabel('Summary Evaluation Latency (ms, log scale)')
ax2.set_title(
    '(B) Relative Binning Summary Evaluation\n'
    'Sequential vs Batched Tensor Broadcast'
)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

ax2.annotate(
    f'Peak: {relbin_speedup[-1]:.1f}x Speedup\n'
    f'({relbin_bat_ms[-1]:.2f} ms for {b_relbin[-1]:,} samples)',
    xy=(b_relbin[-1], relbin_bat_ms[-1]),
    xytext=(300, 0.03),
    arrowprops=dict(arrowstyle='->', color=c_tcuda, lw=1.5),
    bbox=dict(
        boxstyle='round,pad=0.4', facecolor='#E6FFFA',
        edgecolor=c_tcuda, alpha=0.9
    ),
    fontsize=9.5, fontweight='bold', color='#234E52'
)

plt.tight_layout()
save_fig(fig, 'torch_inference_acceleration.png')
plt.close()

# ==============================================================================
# FIGURE 5: Single-Template Production Matched Filtering Scaling
# (MatchedFilterControl.full_matched_filter_and_cluster_symm)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

mfc_json = _find_artifact(
    "matched-filter benchmark artifact",
    "matched_filter_symm_benchmark.json",
    "len_matched_filter_symm_benchmark.json",
)
try:
    with open(mfc_json, "r", encoding="utf-8") as artifact_file:
        mfc_data = json.load(artifact_file)
    mfc_results = mfc_data["campaigns"][0]["results"]
    required_sizes = [32768, 65536, 131072, 262144, 524288]
    if any(f"size_{size}" not in mfc_results for size in required_sizes):
        raise ValueError("results do not contain every segment size")
    for size in required_sizes:
        routes = mfc_results[f"size_{size}"]
        if any(not routes.get(route) for route in (
                "standard_1t", "standard_16t", "torch_cpu_16t",
                "torch_cuda_eager", "torch_cuda")):
            raise ValueError(f"size_{size} is missing a required route")

    mfc_sizes = np.array(required_sizes)
    mfc_labels = [f"{size // 1024}k ({size // 2048}s)"
                  for size in mfc_sizes]
    mfc_std_1t = np.array([
        _to_ms(mfc_results[f"size_{size}"]["standard_1t"])
        for size in mfc_sizes
    ])
    mfc_std_16t = np.array([
        _to_ms(mfc_results[f"size_{size}"]["standard_16t"])
        for size in mfc_sizes
    ])
    mfc_tcpu_16t = np.array([
        _to_ms(mfc_results[f"size_{size}"]["torch_cpu_16t"])
        for size in mfc_sizes
    ])
    mfc_cuda = np.array([
        _to_ms(mfc_results[f"size_{size}"]["torch_cuda_eager"])
        for size in mfc_sizes
    ])
    mfc_cuda_graph = np.array([
        _to_ms(mfc_results[f"size_{size}"]["torch_cuda"])
        for size in mfc_sizes
    ])
except (IndexError, KeyError, TypeError, ValueError,
        json.JSONDecodeError) as exc:
    parser.error(f"invalid matched-filter artifact {mfc_json}: {exc}")

print(f"Loaded matched-filter scaling from {mfc_json}")

mfc_cuda_speedup = mfc_std_1t / mfc_cuda
mfc_tcpu_speedup = mfc_std_1t / mfc_tcpu_16t

# Panel 1: Single-Template Latency vs Segment Length
ax1.plot(
    mfc_sizes, mfc_std_1t, 'd--', color=c_std, linewidth=1.8,
    label='Standard CPU (1T Reference)'
)
ax1.plot(
    mfc_sizes, mfc_std_16t, 'v-.', color='#718096', linewidth=1.4,
    label='Standard CPU (16T FFTW)*'
)
ax1.plot(
    mfc_sizes, mfc_tcpu_16t, 's-', color=c_tcpu, linewidth=1.8,
    label='Torch CPU (16T MKL/SIMD)'
)
ax1.plot(
    mfc_sizes, mfc_cuda, 'o-', color=c_tcuda, linewidth=2.4,
    label=f'{cuda_label} eager'
)
ax1.plot(
    mfc_sizes, mfc_cuda_graph, '*-', color='#2CA02C', linewidth=2.2,
    label='Torch CUDA (CUDA Graph)'
)

ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_ylim(0.05, 40.0)
ax1.set_xticks(mfc_sizes)
ax1.set_xticklabels(mfc_labels, fontsize=9)
ax1.set_xlabel('Segment Length ($N$ samples / duration $T$)')
ax1.set_ylabel('Sequential Latency per Template (ms, log scale)')
ax1.set_title(
    '(A) Single-Template Production Matched Filtering Latency\n'
    '(Correlate + IFFT + Symmetric Peak Clustering)'
)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.text(
    0.03, 0.04,
    "* Std CPU 16T reflects FFTW thread barrier overhead on single 1D FFTs",
    transform=ax1.transAxes, fontsize=8.0, style="italic", color="#555555"
)

ax1.annotate(
    f'Torch CUDA: {np.median(mfc_cuda):.2f} ms median latency\n'
    '(Measured eager execution)',
    xy=(mfc_sizes[2], mfc_cuda[2]),
    xytext=(50000, 1.8),
    arrowprops=dict(arrowstyle='->', color=c_tcuda, lw=1.5),
    bbox=dict(
        boxstyle='round,pad=0.4', facecolor='#E6FFFA',
        edgecolor=c_tcuda, alpha=0.9
    ),
    fontsize=9.5, fontweight='bold', color='#234E52'
)

# Panel 2: Acceleration Multiplier relative to Standard 1T Baseline
ax2.axhline(
    1.0, color=c_std, linestyle='--', linewidth=1.5,
    label='Standard CPU 1T Baseline (1.0x)'
)
ax2.plot(
    mfc_sizes, mfc_cuda_speedup, 'o-', color=c_tcuda, linewidth=2.4,
    label='Torch CUDA Speedup vs 1T'
)
ax2.plot(
    mfc_sizes, mfc_tcpu_speedup, 's-', color=c_tcpu, linewidth=1.8,
    label='Torch CPU 16T Speedup vs 1T'
)

ax2.set_xscale('log', base=2)
ax2.set_ylim(0, 11.0)
ax2.set_xticks(mfc_sizes)
ax2.set_xticklabels(mfc_labels, fontsize=9)
ax2.set_xlabel('Segment Length ($N$ samples / duration $T$)')
ax2.set_ylabel('Speedup Factor vs 1-Thread CPU')
ax2.set_title(
    '(B) Single-Template Acceleration Multiplier vs Segment Length\n'
    f'(Peak measured speedup: {np.max(mfc_cuda_speedup):.2f}x)'
)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

max_mfc_idx = np.argmax(mfc_cuda_speedup)
ax2.annotate(
    f'Peak: {mfc_cuda_speedup[max_mfc_idx]:.2f}x Speedup\n'
    f'({mfc_cuda[max_mfc_idx]:.2f} ms vs '
    f'{mfc_std_1t[max_mfc_idx]:.2f} ms at '
    f'{mfc_sizes[max_mfc_idx] // 2048}s)',
    xy=(mfc_sizes[max_mfc_idx], mfc_cuda_speedup[max_mfc_idx]),
    xytext=(100000, 6.5),
    arrowprops=dict(arrowstyle='->', color=c_tcuda, lw=1.5),
    bbox=dict(
        boxstyle='round,pad=0.4', facecolor='#E6FFFA',
        edgecolor=c_tcuda, alpha=0.9
    ),
    fontsize=9.5, fontweight='bold', color='#234E52'
)

plt.tight_layout()
save_fig(fig, 'torch_matched_filter_symm_scaling.png')
plt.close()

# ==============================================================================
# FIGURE 6: CPU Multi-Thread Scaling & Component Breakdown (1T-64T)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

cpu_threads = np.array([1, 4, 8, 16, 32, 64])
cpu_sizes = [32768, 65536, 131072, 262144, 524288]
cpu_size_labels = ['32k (16s)', '65k (32s)', '131k (64s)', '262k (128s)', '524k (256s)']

cpu_breakdown_json = _find_artifact(
    "CPU profile artifact", "cpu_profile_breakdown.json"
)
try:
    with open(cpu_breakdown_json, "r", encoding="utf-8") as artifact_file:
        cpu_profile = json.load(artifact_file)
    scaling_by_size = {size: {} for size in cpu_sizes}
    components = {name: {} for name in ("corr_us", "ifft_us", "cluster_us")}
    for item in cpu_profile:
        size = item.get("N")
        thread_count = item.get("threads")
        if size in scaling_by_size and thread_count in cpu_threads:
            scaling_by_size[size][thread_count] = item["full_us"]["median"] / 1000.0
            if size == cpu_sizes[-1]:
                for name in components:
                    components[name][thread_count] = item[name]["median"] / 1000.0
    if any(any(thread not in scaling_by_size[size] for thread in cpu_threads)
           for size in cpu_sizes):
        raise ValueError("profile does not contain every size/thread cell")
    if any(any(thread not in values for thread in cpu_threads)
           for values in components.values()):
        raise ValueError("profile does not contain the complete component data")
    cpu_scaling_data = {
        size: [scaling_by_size[size][thread] for thread in cpu_threads]
        for size in cpu_sizes
    }
    comp_corr = [components["corr_us"][thread] for thread in cpu_threads]
    comp_ifft = [components["ifft_us"][thread] for thread in cpu_threads]
    comp_clust = [components["cluster_us"][thread] for thread in cpu_threads]
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    parser.error(f"invalid CPU profile artifact {cpu_breakdown_json}: {exc}")

print(f"Loaded CPU scaling breakdown from {cpu_breakdown_json}")

# Panel 1: End-to-End Latency vs Thread Count
colors_n = ['#718096', '#3182CE', '#38A169', '#D69E2E', '#E53E3E']
markers_n = ['o', 's', '^', 'D', 'v']

for idx, n_val in enumerate(cpu_sizes):
    ax1.plot(
        cpu_threads, cpu_scaling_data[n_val],
        marker=markers_n[idx], color=colors_n[idx], linewidth=1.8,
        label=f'N = {cpu_size_labels[idx]}'
    )

# Highlight sweet spot (16 threads)
ax1.axvspan(12, 20, color='#38A169', alpha=0.15, label='Optimal Bandwidth Range (16T)')

ax1.set_xlabel('CPU Threads (OMP_NUM_THREADS / MKL_NUM_THREADS)')
ax1.set_ylabel('Sequential Execution Latency (ms, log scale)')
ax1.set_title('(A) CPU Thread Scaling vs Segment Size $N$\n(MKL DFTI + SIMD Peak Clustering)')
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xticks(cpu_threads)
ax1.set_xticklabels([f'{t}T' for t in cpu_threads])
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right', fontsize=8.5)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# Panel 2: Stacked Operation Breakdown at N=524288
x_indices = np.arange(len(cpu_threads))
width_b = 0.55

bar_corr = ax2.bar(
    x_indices, comp_corr, width_b,
    label='Frequency Correlation (Htilde * Stilde)',
    color='#63B3ED', edgecolor='black', linewidth=0.5
)
bar_ifft = ax2.bar(
    x_indices, comp_ifft, width_b, bottom=comp_corr,
    label='MKL DFTI Inverse FFT (IFFT)',
    color='#3182CE', edgecolor='black', linewidth=0.5
)
bar_clust = ax2.bar(
    x_indices, comp_clust, width_b,
    bottom=np.array(comp_corr) + np.array(comp_ifft),
    label='Peak Finding & Clustering',
    color='#DD6B20', edgecolor='black', linewidth=0.5
)

# Annotate speedup at 16T
speedup_16t = cpu_scaling_data[524288][0] / cpu_scaling_data[524288][3]
ax2.annotate(
    f'16T Optimum: {cpu_scaling_data[524288][3]:.2f} ms\n({speedup_16t:.1f}x speedup vs 1T)',
    xy=(3, cpu_scaling_data[524288][3]),
    xytext=(1.8, 3.2),
    arrowprops=dict(arrowstyle='->', color='#2B6CB0', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF8FF', edgecolor='#3182CE', alpha=0.9),
    fontsize=9, fontweight='bold', color='#2B6CB0'
)

# Annotate cross-NUMA barrier penalty at 64T
ax2.annotate(
    '64T Cross-NUMA Barrier\nOverhead Penalty',
    xy=(5, cpu_scaling_data[524288][5]),
    xytext=(3.5, 4.2),
    arrowprops=dict(arrowstyle='->', color='#C53030', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5F5', edgecolor='#E53E3E', alpha=0.9),
    fontsize=8.5, fontweight='bold', color='#9B2C2C'
)

ax2.set_xlabel('CPU Threads')
ax2.set_ylabel('Execution Time (ms) [N = 524,288 / 256s]')
ax2.set_title('(B) Component Scaling & Amdahl Bottlenecks ($N=524,288$)\nLinear MKL Scaling up to 16 Threads')
ax2.set_xticks(x_indices)
ax2.set_xticklabels([f'{t}T' for t in cpu_threads])
ax2.set_ylim(0, 5.2)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right', fontsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
save_fig(fig, 'torch_cpu_thread_scaling.png')
plt.close()
