#!/usr/bin/env python3
"""Reproduce chi-square diagnostic tables, figures and an offline HTML report.

This reads captured inputs and outputs; it does not run a search, import PyCBC,
alter a scientific tolerance, or measure performance. See the report for the
distinction between the compiled CPU observations and the phase-only model.
"""

import argparse
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fft import ifft


RTOL = 1e-4
ATOL = 1e-5
ARMS = ("original_cpu", "branch_cpu", "jax_cpu_batched", "jax_cuda_batched")
COLORS = ("#a6681b", "#595959", "#2166ac", "#168273")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(reference, value):
    reference, value = np.asarray(reference), np.asarray(value)
    delta = np.abs(value - reference)
    return {
        "n": len(reference),
        "max_absolute_difference": float(np.max(delta)),
        "max_relative_difference": float(np.max(delta / np.abs(reference))),
        "outside_tolerance": int(np.sum(delta > ATOL + RTOL * np.abs(reference))),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trigger_values(path):
    with h5py.File(path) as file:
        # The campaign uses integral sample indices relative to this GPS start.
        times = np.rint((file["H1/end_time"][:] - 1187007048) * 2048).astype(int)
        keys = list(zip(file["H1/template_hash"][:].tolist(), times.tolist()))
        values = file["H1/chisq"][:].astype(np.float64)
    assert len(set(keys)) == len(keys), "Repeated trigger identity"
    return dict(zip(keys, values))


def retained_points(evidence, observed):
    lookup = {}
    for call_id, call in enumerate(observed, 1):
        for index, sample in enumerate(call["time_indices"]):
            key = (call["template_hash"], sample)
            assert key not in lookup
            lookup[key] = {
                "call": call_id, "point_in_call": index,
                "local_sample": call["indices"][index],
                "max_bin_width": int(np.max(np.diff(call["bins"]))),
                "observed_cpu": call["reference"][index],
                "direct_cpu_inputs": call["accurate"][index],
                "cpu_double": call["cpu_double"][index],
            }
    arms = {arm: trigger_values(evidence / "qualification" / arm / "triggers.hdf")
            for arm in ARMS}
    keys = sorted(arms["original_cpu"])
    assert all(set(arm) == set(keys) for arm in arms.values())
    rows = []
    for key in keys:
        values = {arm: arms[arm][key] for arm in ARMS}
        assert values["original_cpu"] == lookup[key]["observed_cpu"]
        assert values["branch_cpu"] == values["original_cpu"]
        rows.append({"template_hash": key[0], "sample_from_gps_start": key[1],
                     **lookup[key], **values})
    return rows


def fft_points(evidence, observed):
    rows = []
    for path in sorted(evidence.glob("point-input-*.npz")):
        call = observed[int(path.stem.rsplit("-", 1)[1]) - 1]
        with np.load(path) as data:
            corr, points, bins = data["corr"], data["indices"], data["bins"]
            assert corr.dtype == np.complex64
            n_time = len(corr)
            power = np.zeros(len(points), dtype=np.float64)
            truncated_power = power.copy()
            direct_power = power.copy()
            for lo, hi in zip(bins[:-1], bins[1:]):
                band = np.zeros(n_time, dtype=np.complex128)
                band[lo:hi] = corr[lo:hi]
                # SciPy's inverse FFT includes 1/N; PyCBC's bin sum does not.
                amplitude = (ifft(band, workers=1) * n_time)[points]
                power += np.abs(amplitude) ** 2
                frequencies = np.arange(lo, hi, dtype=np.float64)
                integer_frequencies = np.arange(lo, hi, dtype=np.int64)
                for index, point in enumerate(points):
                    phase = np.exp(2j * 3.141592653 * float(point)
                                   * frequencies / n_time)
                    amplitude = np.sum(corr[lo:hi] * phase, dtype=np.complex128)
                    truncated_power[index] += abs(amplitude) ** 2
                    cycles = (integer_frequencies * int(point)) % n_time
                    phase = np.exp(2j * np.pi * cycles / n_time)
                    amplitude = np.sum(corr[lo:hi] * phase, dtype=np.complex128)
                    direct_power[index] += abs(amplitude) ** 2
            # Hold the saved complex64 SNR term and normalization fixed.
            snr_power = (data["snr"].conj() * data["snr"]).real
            norm2 = float(data["norm"]) ** 2
            fft = (power * (len(bins) - 1) - snr_power) * norm2
            truncated = (truncated_power * (len(bins) - 1) - snr_power) * norm2
            direct = (direct_power * (len(bins) - 1) - snr_power) * norm2
            for i, point in enumerate(points):
                rows.append({
                    "input": path.name, "point_in_call": i,
                    "local_sample": int(point), "ntime": n_time,
                    "template_hash": call["template_hash"],
                    "sample_from_gps_start": call["time_indices"][i],
                    "fft": float(fft[i]), "original_cpu": call["reference"][i],
                    "direct_cpu_inputs": call["accurate"][i],
                    "cpu_double": call["cpu_double"][i],
                    "truncated_pi_direct": float(truncated[i]),
                    "direct_complex128": float(direct[i]),
                })
    assert len(rows) == 11, "Expected six captured inputs / eleven point checks"
    return rows


def phase_trace(evidence, example):
    """Explicit rounded phase model, not a measurement inside compiled Cython."""
    with np.load(evidence / example["input"]) as data:
        n_time, point = len(data["corr"]), example["local_sample"]
        bins = data["bins"]
        index = int(np.argmax(np.diff(bins)))
        start, end = int(bins[index]), int(bins[index + 1])
    angle = 2 * 3.141592653 * point / n_time
    initial = 2 * 3.141592653 * point * start / n_time
    pr, pi = np.float32(math.cos(initial)), np.float32(math.sin(initial))
    rr, ri = np.float32(math.cos(angle)), np.float32(math.sin(angle))
    double_states = {}
    for name, pi_constant in [("double_cpu_pi", 3.141592653),
                              ("double_full_pi", math.pi)]:
        a = 2 * pi_constant * point / n_time
        initial_a = 2 * pi_constant * point * start / n_time
        double_states[name] = [math.cos(initial_a), math.sin(initial_a),
                               math.cos(a), math.sin(a)]
    rows = []
    for step in range(end - start):
        if step % 512 == 0 or step == end - start - 1:
            exact = np.exp(2j * np.pi * (((start + step) * point) % n_time) / n_time)
            direct = np.complex64(exact)
            recurrence = complex(pr, pi)
            row = {
                "steps": step, "frequency_index": start + step,
                "recurrence_modulus_minus_one": abs(recurrence) - 1,
                "direct_modulus_minus_one": abs(complex(direct)) - 1,
                "recurrence_phase_error_rad": float(np.angle(recurrence / exact)),
                "direct_phase_error_rad": float(np.angle(direct / exact)),
                "pi_constant_phase_bias_rad": (
                    2 * (3.141592653 - math.pi) * point * (start + step) / n_time),
            }
            for name, (dr, di, _, _) in double_states.items():
                value = complex(dr, di)
                row[name + "_modulus_minus_one"] = abs(value) - 1
                row[name + "_phase_error_rad"] = float(np.angle(value / exact))
            rows.append(row)
        t1, t2 = pr, pi
        # Separate float32 operations, with no fused multiply-add.
        pr = np.float32(np.float32(t1 * rr) - np.float32(t2 * ri))
        pi = np.float32(np.float32(t1 * ri) + np.float32(t2 * rr))
        for state in double_states.values():
            dr, di, drr, dri = state
            # Python float operations round to float64 without contraction.
            state[0] = dr * drr - di * dri
            state[1] = dr * dri + di * drr
    return rows, {"input": example["input"], "point": point, "ntime": n_time,
                  "bin_index_zero_based": index, "start": start, "end": end,
                  "width": end - start, "rounded_rotation_modulus": abs(complex(rr, ri)),
                  "model": "Separate float32/float64 operations; not compiled CPU instrumentation"}


def save_figure(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, facecolor="white")
    fig.savefig(output / (name + ".svg"), metadata={"Date": None})
    plt.close(fig)


def figures(output, retained, fft, phase, phase_info):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "pycbc-chisq"})
    cpu = np.array([r["original_cpu"] for r in retained])
    direct = np.array([r["direct_cpu_inputs"] for r in retained])
    order = np.argsort(np.abs(direct - cpu) / (ATOL + RTOL * np.abs(cpu)))
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), layout="constrained")
    for arm, label, color, marker, size in [
            ("direct_cpu_inputs", "Direct sums on original inputs", COLORS[1], "o", 7),
            ("jax_cpu_batched", "Saved JAX CPU", COLORS[2], "+", 6),
            ("jax_cuda_batched", "Saved JAX CUDA", COLORS[3], ".", 3)]:
        value = np.array([r[arm] for r in retained])
        scaled = np.abs(value - cpu) / (ATOL + RTOL * np.abs(cpu))
        axes[0].plot(np.arange(1, len(cpu) + 1), scaled[order], marker, ms=size,
                     markerfacecolor="none" if marker == "o" else color,
                     color=color, label=label)
    axes[0].axhline(1, color="#333333", ls="--", lw=1, label="Compatibility boundary")
    axes[0].set(ylabel="Difference / allowed difference", ylim=(0, 21),
                title="A. Compatibility with the original CPU: 77 retained triggers")
    axes[0].legend(fontsize=8, ncol=2)
    for arm, label, color in [("original_cpu", "Original CPU", COLORS[0]),
                              ("jax_cpu_batched", "Saved JAX CPU", COLORS[2]),
                              ("jax_cuda_batched", "Saved JAX CUDA", COLORS[3])]:
        value = np.array([r[arm] for r in retained])
        relative = np.abs(value - direct) / np.abs(direct)
        nonzero = relative[order] > 0
        axes[1].semilogy(np.arange(1, len(cpu) + 1)[nonzero], relative[order][nonzero],
                         ".", color=color, label=label, ms=5)
    axes[1].set_ylim(3e-8, 4e-3)
    axes[1].set(xlabel="Trigger rank, sorted by panel A's controlled difference",
                ylabel="Relative difference from direct sums",
                title="B. Same outputs compared with direct sums on the original inputs")
    axes[1].legend(fontsize=8, loc="upper left", ncol=3)
    for ax in axes:
        ax.grid(alpha=.16)
    save_figure(fig, output, "retained_triggers")

    oracle = np.array([r["fft"] for r in fft])
    fig, ax = plt.subplots(figsize=(9.5, 4.5), layout="constrained")
    for field, label, color, marker in [
            ("original_cpu", "Original CPU (complex64)", COLORS[0], "o"),
            ("cpu_double", "CPU recurrence with complex128 control inputs", COLORS[1], "s"),
            ("truncated_pi_direct", "Direct sums with CPU's pi constant", "#936bb0", "x"),
            ("direct_cpu_inputs", "Direct sums, float32 output", COLORS[2], "^"),
            ("direct_complex128", "Direct sums, float64 output", COLORS[3], "d")]:
        values = np.array([r[field] for r in fft])
        relative = np.abs(values - oracle) / np.abs(oracle)
        relative[relative == 0] = np.nan  # Exact agreement has no log ordinate.
        ax.semilogy(range(1, len(fft) + 1), relative,
                    marker, label=label, color=color, ms=6)
    ax.plot(range(1, len(fft) + 1), RTOL + ATOL / np.abs(oracle), "--", color="#333333",
            label="Existing tolerance, shown against FFT for scale")
    ax.set(xticks=range(1, len(fft) + 1), xlabel="Captured point (first six calls; before NewSNR cut)",
           ylabel="Relative difference from double-precision FFT",
           title="Independent inverse-FFT check: 11 points from six actual correlations")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.20))
    ax.grid(alpha=.16)
    save_figure(fig, output, "fft_controls")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6), layout="constrained", sharex=True)
    steps = [r["steps"] for r in phase]
    for field, label, color in [("recurrence", "Explicit float32 recurrence model", COLORS[0]),
                                ("direct", "Independent phase, cast to complex64", COLORS[2])]:
        axes[0].plot(steps, [r[field + "_modulus_minus_one"] for r in phase],
                     label=label, color=color)
        axes[1].plot(steps, [r[field + "_phase_error_rad"] for r in phase], color=color)
    axes[0].set(ylabel=r"Phase modulus minus one: $|q_k|-1$",
                title=f"Phase-only illustration: sample {phase_info['point']}, "
                      f"a {phase_info['width']:,}-sample frequency bin")
    axes[0].legend(fontsize=9)
    axes[1].set(ylabel="Wrapped phase difference (radians)",
                xlabel="Frequency steps since the beginning of this bin")
    for ax in axes:
        ax.grid(alpha=.16)
    save_figure(fig, output, "phase_recurrence")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6), layout="constrained", sharex=True)
    for field, label, color in [
            ("recurrence", "float32, original pi constant", COLORS[0]),
            ("double_cpu_pi", "float64, original pi constant", COLORS[1]),
            ("double_full_pi", "float64, full double-precision pi", COLORS[2])]:
        for ax, suffix in zip(axes, ["_modulus_minus_one", "_phase_error_rad"]):
            value = np.abs([r[field + suffix] for r in phase])
            value[value == 0] = np.nan
            ax.semilogy(steps, value, label=label, color=color)
    axes[1].semilogy(steps, np.abs([r["pi_constant_phase_bias_rad"] for r in phase]),
                     "--", color="#936bb0", label="Phase bias from pi constant alone")
    axes[0].set(ylabel=r"Absolute modulus error: $||q_k|-1|$",
                title="Precision controls on the same phase recurrence (explicit models)")
    axes[0].legend(fontsize=8, loc="center left")
    axes[1].set(ylabel="Absolute phase difference (radians)",
                xlabel="Frequency steps since the beginning of this bin")
    axes[1].legend(fontsize=8, loc="center left")
    for ax in axes:
        ax.grid(alpha=.16)
    save_figure(fig, output, "phase_precision")


def source_pages(evidence, output):
    target = output / "source"
    target.mkdir(exist_ok=True)
    for path in sorted((evidence / "source").iterdir()):
        if not path.is_file():
            continue
        (target / path.name).write_bytes(path.read_bytes())
        lines = "\n".join(f'<span id="L{i}">{i:4d}  {html.escape(line)}</span>'
                          for i, line in enumerate(path.read_text().splitlines(), 1))
        (target / (path.name + ".html")).write_text(
            '<!doctype html><meta charset="utf-8"><title>' + html.escape(path.name)
            + '</title><style>body{margin:30px}pre{font-size:13px;line-height:1.5}'
            'span:target{background:#fff1bf}</style><h1>' + html.escape(path.name)
            + '</h1><p>Source snapshot; SHA-256 ' + digest(path) + '</p><pre>' + lines + '</pre>')


def render_report(rst, output):
    from docutils.core import publish_string
    text = rst.read_text()
    # Sphinx cross-reference target is not needed in standalone docutils output.
    text = re.sub(r"^\.\. _jax-chisq-numerics:\n\n", "", text)
    prefix = "_static/jax_chisq_numerics/"
    text = text.replace(prefix, str(output.resolve()) + "/")
    document = publish_string(text, writer_name="html5", settings_overrides={
        "math_output": "MathML", "initial_header_level": 2,
        "report_level": 2, "halt_level": 2, "syntax_highlight": "short",
        "embed_stylesheet": True,
    }).decode("utf-8")
    for path in output.glob("*.png"):
        document = document.replace(str(path.resolve()), "data:image/png;base64,"
                                    + base64.b64encode(path.read_bytes()).decode())
    appendices = []
    for path in sorted((output / "source").glob("*.html")):
        source = path.read_text().split("<pre>", 1)[1].split("</pre>", 1)[0]
        source = re.sub(r'id="L(\d+)"', rf'id="source-{path.name}-L\1"', source)
        document = document.replace(str(path.resolve()) + "#L", "#source-" + path.name + "-L")
        appendices.append('<details><summary>Source snapshot: ' + path.stem
                          + '</summary><pre>' + source + '</pre></details>')
    for name in ["retained_triggers.csv", "fft_controls.csv", "phase_recurrence.csv",
                 "summary.json", "manifest.json"]:
        path = output / name
        mime = "text/csv" if path.suffix == ".csv" else "application/json"
        uri = "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()
        document = document.replace('href="' + str(path.resolve()) + '"',
                                    'download="' + name + '" href="' + uri + '"')
    style = """<style>
body{max-width:1000px;margin:48px auto;padding:0 28px;color:#24313d;
font-family:system-ui,sans-serif;font-size:16px;line-height:1.65}
h1,h2,h3{color:#17384d;line-height:1.25}h1{font-size:2.3rem}h2{margin-top:2.4rem}
table{border-collapse:collapse;font-size:.91rem;width:100%;margin:1.4rem 0}
td,th{padding:9px 12px;border-bottom:1px solid #d8e0e5;text-align:left}
th{background:#edf3f6}img{max-width:100%;height:auto}pre{background:#f4f6f8;
padding:16px;overflow-x:auto;font-size:.8rem;line-height:1.45}a{color:#175e88}
.caption{font-size:.93rem;color:#46586b}.math{overflow-x:auto;padding:12px}
details{margin:12px 0}span:target{background:#fff1bf}
@media print{body{font-size:10pt;margin:0}h2{break-after:avoid}figure,table{break-inside:avoid}}
</style>"""
    # Opening a source anchor also opens its collapsed source panel.
    script = """<script>function reveal(){let e=document.getElementById(
decodeURIComponent(location.hash.slice(1)));if(e){let d=e.closest('details');
if(d){d.open=true;e.scrollIntoView();}}}addEventListener('hashchange',reveal);reveal();</script>"""
    document = document.replace("</head>", style + "</head>").replace(
        "</body>", '<h2>Referenced source snapshots</h2>' + "".join(appendices) + script + "</body>")
    (output / "report.html").write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", type=Path, help="RST report to render as offline HTML")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    observed = read_json(args.evidence / "points.json")
    retained = retained_points(args.evidence, observed)
    fft = fft_points(args.evidence, observed)
    retained_keys = {(r["template_hash"], r["sample_from_gps_start"]) for r in retained}
    for row in fft:
        row["retained_after_newsnr"] = (row["template_hash"], row["sample_from_gps_start"]) in retained_keys
    example = max(fft, key=lambda r: abs(r["original_cpu"] - r["fft"]) / abs(r["fft"]))
    phase, phase_info = phase_trace(args.evidence, example)
    for name, rows in [("retained_triggers", retained), ("fft_controls", fft),
                       ("phase_recurrence", phase)]:
        write_csv(output / (name + ".csv"), rows)
    summary = {
        "scope": "Diagnostic arithmetic comparisons; not performance measurements",
        "rtol": RTOL, "atol": ATOL,
        "retained_count": len(retained), "observed_count": sum(len(r["indices"]) for r in observed),
        "observed_calls": len(observed), "max_observed_bin_width": max(
            int(np.max(np.diff(r["bins"]))) for r in observed),
        "original_cpu_replay_exact": True, "branch_cpu_exact": True,
        "compatibility_reference_original_cpu": {
            arm: metrics([r["original_cpu"] for r in retained], [r[arm] for r in retained])
            for arm in ("direct_cpu_inputs", "jax_cpu_batched", "jax_cuda_batched", "cpu_double")},
        "direct_sum_reference_original_cpu_inputs": {
            arm: metrics([r["direct_cpu_inputs"] for r in retained], [r[arm] for r in retained])
            for arm in ("original_cpu", "jax_cpu_batched", "jax_cuda_batched", "cpu_double")},
        "fft_reference": {field: metrics([r["fft"] for r in fft], [r[field] for r in fft])
                          for field in ("original_cpu", "direct_cpu_inputs", "cpu_double",
                                        "truncated_pi_direct", "direct_complex128")},
        "cpu_double_vs_truncated_pi_direct": metrics(
            [r["truncated_pi_direct"] for r in fft], [r["cpu_double"] for r in fft]),
        "example": example, "phase_model": phase_info, "phase_model_last_sample": phase[-1],
        "plot_runtime": {"python": sys.version, "numpy": np.__version__,
                         "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                         "h5py": h5py.__version__},
    }
    saved = read_json(args.evidence / "analysis.json")
    summary["complex128_controls"] = {
        "scope": "Captured complex64 correlations promoted to complex128; saved SNR term and normalization fixed",
        "ratio_of_maximum_relative_errors_cpu64_to_cpu128": (
            summary["fft_reference"]["original_cpu"]["max_relative_difference"]
            / summary["fft_reference"]["cpu_double"]["max_relative_difference"]),
        "direct_complex128_exact_fft_matches": sum(r["direct_complex128"] == r["fft"] for r in fft),
    }
    np.testing.assert_allclose(
        [r["direct_complex128"] for r in fft], [r["fft"] for r in fft], rtol=1e-11, atol=1e-11)
    for group, old in [("compatibility_reference_original_cpu", "cpu_double_vs_original"),
                       ("direct_sum_reference_original_cpu_inputs", "cpu_double_vs_accurate")]:
        current, prior = summary[group]["cpu_double"], saved["comparisons"][old]
        np.testing.assert_allclose(current["max_relative_difference"],
                                   prior["max_relative_difference"], rtol=1e-12)
        assert current["outside_tolerance"] == prior["failed"]
    for field, old in [("direct_cpu_inputs", "accurate_vs_original"),
                       ("jax_cpu_batched", "jax_cpu_batched_vs_accurate_cpu_inputs"),
                       ("jax_cuda_batched", "jax_cuda_batched_vs_accurate_cpu_inputs")]:
        group = "compatibility_reference_original_cpu" if field == "direct_cpu_inputs" else "direct_sum_reference_original_cpu_inputs"
        np.testing.assert_allclose(summary[group][field]["max_relative_difference"],
                                   saved["comparisons"][old]["max_relative_difference"], rtol=1e-12)
    for field, old in [("original_cpu", "upstream_vs_fft"),
                       ("direct_cpu_inputs", "accurate_vs_fft"),
                       ("truncated_pi_direct", "truncated_pi_direct_vs_fft")]:
        prior = max(r[old]["max_relative_difference"] for r in saved["fft_checks"])
        np.testing.assert_allclose(summary["fft_reference"][field]["max_relative_difference"], prior,
                                   rtol=1e-5, atol=1e-12)
    write_json(output / "summary.json", summary)
    files = [p for p in args.evidence.rglob("*") if p.is_file() and (
        p.suffix in (".npz", ".hdf", ".json") or p.parent.name == "source")]
    write_json(output / "manifest.json", {"input_sha256": {
        str(path.relative_to(args.evidence)): digest(path) for path in sorted(files)},
        "generator_sha256": digest(Path(__file__))})
    figures(output, retained, fft, phase, phase_info)
    source_pages(args.evidence, output)
    if args.render:
        render_report(args.render, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
