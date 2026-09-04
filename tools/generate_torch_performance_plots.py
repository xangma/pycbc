#!/usr/bin/env python3
"""Render Torch performance evidence without overstating benchmark scope.

Current production-live-batch artifacts and older comprehensive-suite JSON are
both accepted. Public-library, private-component, and legacy/unknown routing
are labelled distinctly, and uncertainty comes from each summary's retained
``median_ci95`` (or its raw samples for legacy summaries).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from tools.benchmark_artifact import (
        atomic_write_json,
        summary_median_ci95,
        summary_statistic,
    )
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from benchmark_artifact import (
        atomic_write_json,
        summary_median_ci95,
        summary_statistic,
    )


LIVE_ARTIFACT_NAMES = (
    "production_live_batch.json",
    "len_production_live_batch.json",
    "live_batch_latest.json",
    "extended_full_suite.json",
    "comprehensive_benchmark_results.json",
    "len_extended_full_suite.json",
)
MATCHED_FILTER_ARTIFACT_NAMES = (
    "matched_filter_symm_benchmark.json",
    "len_matched_filter_symm_benchmark.json",
)

ROUTE_ALIASES = {
    "original_standard": ("original_standard", "original_1t"),
    "branch_standard": (
        "branch_standard",
        "original_16t",
        "original_threaded_16t",
    ),
    "torch_cpu": ("torch_cpu", "torch_cpu_16t"),
    "torch_cpu_native": ("torch_cpu_native",),
    "torch_cuda": ("torch_cuda",),
    "torch_cuda_native": ("torch_cuda_native",),
}

ROUTE_STYLE = {
    "original_standard": ("#4A5568", "d", "--"),
    "branch_standard": ("#718096", "v", "-."),
    "torch_cpu": ("#3182CE", "s", "-"),
    "torch_cpu_native": ("#00A3C4", "P", "--"),
    "torch_cuda": ("#38A169", "o", "-"),
    "torch_cuda_native": ("#805AD5", "X", "--"),
}


def _median(obj: Mapping) -> float:
    """Backward-compatible median accessor retained for plot consumers."""

    return summary_statistic(obj, "p50")


def _to_ms(obj: Mapping) -> float:
    """Convert a historical seconds summary's median to milliseconds."""

    return _median(obj) * 1000.0


def _to_tput(batch: int, obj: Mapping) -> float:
    """Convert a historical seconds-per-batch summary to item throughput."""

    return float(batch) / _median(obj)


def _get_route(result: Mapping, *names: str) -> Mapping:
    for name in names:
        if name in result and result[name]:
            return result[name]
    raise ValueError(f"benchmark result is missing routes {names!r}")


def _optional_stat(summary: Mapping, statistic: str) -> float | None:
    try:
        return summary_statistic(summary, statistic)
    except (KeyError, TypeError, ValueError):
        return None


def _normalized_summary(
    summary: Mapping,
    *,
    scale: float = 1.0,
    unit: str,
) -> dict:
    """Normalize current and legacy summaries without losing raw samples."""

    point = summary_statistic(summary, "p50")
    low, high = summary_median_ci95(summary)
    samples = summary.get("samples", [])
    interval = summary.get("median_ci95")
    uncertainty_available = bool(samples) or (
        isinstance(interval, Mapping) and {"low", "high"} <= interval.keys()
    )
    normalized = {
        "unit": unit,
        "count": int(summary.get("count", len(samples) or 1)),
        "samples": [float(value) * scale for value in samples],
        "p50": point * scale,
        "p95": None,
        "p99": None,
        "median_ci95": {"low": low * scale, "high": high * scale},
        "median_ci95_available": uncertainty_available,
    }
    for statistic in ("p95", "p99"):
        value = _optional_stat(summary, statistic)
        if value is not None:
            normalized[statistic] = value * scale
    return normalized


def _point_summary(value: float, *, unit: str) -> dict:
    """Represent a legacy point estimate while marking tails unavailable."""

    value = float(value)
    return {
        "unit": unit,
        "count": 1,
        "samples": [],
        "p50": value,
        "p95": None,
        "p99": None,
        "median_ci95": {"low": value, "high": value},
        "median_ci95_available": False,
    }


def _inverse_summary(summary: Mapping, numerator: float, *, unit: str) -> dict:
    """Transform latency to throughput, including the inverse CI bounds."""

    latency = summary_statistic(summary, "p50")
    if latency <= 0:
        raise ValueError("latency measurements must be positive")
    low, high = summary_median_ci95(summary)
    if low <= 0 or high <= 0:
        raise ValueError("latency confidence interval must be positive")
    samples = summary.get("samples", [])
    transformed = [numerator / float(value) for value in samples if value > 0]
    return {
        "unit": unit,
        "count": int(summary.get("count", len(samples) or 1)),
        "samples": transformed,
        "p50": numerator / latency,
        "p95": None,
        "p99": None,
        "median_ci95": {
            "low": numerator / high,
            "high": numerator / low,
        },
        "median_ci95_available": bool(summary.get("samples"))
        or bool(summary.get("median_ci95")),
    }


def _ratio_summary(
    numerator: Mapping,
    denominator: Mapping,
    *,
    unit: str,
) -> dict:
    """Form a conservative ratio interval from two median intervals."""

    numerator_point = float(numerator["p50"])
    denominator_point = float(denominator["p50"])
    if denominator_point <= 0:
        raise ValueError("ratio denominator must be positive")
    numerator_ci = numerator["median_ci95"]
    denominator_ci = denominator["median_ci95"]
    denominator_low = float(denominator_ci["low"])
    denominator_high = float(denominator_ci["high"])
    if denominator_low <= 0 or denominator_high <= 0:
        raise ValueError("ratio denominator interval must be positive")
    return {
        "unit": unit,
        "count": min(int(numerator["count"]), int(denominator["count"])),
        "samples": [],
        "p50": numerator_point / denominator_point,
        "p95": None,
        "p99": None,
        "median_ci95": {
            "low": float(numerator_ci["low"]) / denominator_high,
            "high": float(numerator_ci["high"]) / denominator_low,
        },
        "median_ci95_available": bool(
            numerator.get("median_ci95_available", True)
            and denominator.get("median_ci95_available", True)
        ),
    }


def _summary_from_legacy_fields(cell: Mapping) -> dict:
    latency = cell.get("latency_block_median_ms")
    if latency is None:
        latency = cell.get("latency_block_p50_ms")
    if latency is None:
        raise ValueError("route has no latency summary")
    result = _point_summary(latency, unit="ms/block")
    result["p95"] = cell.get("latency_block_p95_ms")
    result["p99"] = cell.get("latency_block_p99_ms")
    return result


def _legacy_route_definition(route: str) -> dict:
    return {
        "label": route.replace("_", " ").title(),
        "routing_mode": "unknown_legacy",
        "experimental": None,
        "feature_flags": {},
    }


def _route_label(route: str, definition: Mapping) -> str:
    label = str(definition.get("label", route.replace("_", " ").title()))
    mode = definition.get("routing_mode", "unknown_legacy")
    suffix = {
        "production_default": "default",
        "experimental_native": "experimental",
        "unknown_legacy": "legacy routing unknown",
    }.get(mode, str(mode).replace("_", " "))
    return f"{label} [{suffix}]"


def _parse_production_live_batch(data: Mapping) -> dict:
    summary_by_batch = data.get("summary_by_batch")
    if not isinstance(summary_by_batch, Mapping) or not summary_by_batch:
        raise ValueError("production artifact has no summary_by_batch")
    batches = data.get("batches") or sorted(
        int(key.removeprefix("batch_")) for key in summary_by_batch
    )
    batches = [int(batch) for batch in batches]
    first_cell = summary_by_batch[f"batch_{batches[0]}"]
    routes = list(data.get("routes") or first_cell.keys())
    reference_route = data.get("reference_route", routes[0])
    definitions = data.get("route_definitions", {})

    measurement = data.get("measurement")
    if not isinstance(measurement, Mapping):
        # The previous driver timed the private method. State that explicitly
        # instead of promoting those measurements to public-API evidence.
        measurement = {
            "scope": "component_internal",
            "call_surface": "LiveBatchMatchedFilter._process_batch",
            "claim_level": "private_component",
            "full_cli_end_to_end": {"measured": False},
            "limitations": [{"code": "legacy_private_call_surface_inferred"}],
        }

    normalized_routes = {}
    for route in routes:
        definition = definitions.get(route) or _legacy_route_definition(route)
        route_data = {
            "definition": definition,
            "label": _route_label(route, definition),
            "latency_block_ms": [],
            "cold_latency_block_ms": [],
            "latency_per_waveform_ms": [],
            "throughput_wps": [],
            "speedup_vs_reference": [],
            "peak_cuda_allocated_bytes": [],
            "peak_cuda_reserved_bytes": [],
            "peak_cuda_memory_measurement": None,
        }
        for batch in batches:
            cell = summary_by_batch[f"batch_{batch}"][route]
            latency_obj = cell.get("latency_block_ms")
            latency = (
                _normalized_summary(latency_obj, unit="ms/block")
                if isinstance(latency_obj, Mapping)
                else _summary_from_legacy_fields(cell)
            )
            throughput_obj = cell.get("throughput_wps")
            throughput = (
                _normalized_summary(
                    throughput_obj, unit="waveforms/second"
                )
                if isinstance(throughput_obj, Mapping)
                else _point_summary(
                    cell.get(
                        "throughput_wps_median",
                        batch * 1000.0 / latency["p50"],
                    ),
                    unit="waveforms/second",
                )
            )
            per_waveform_obj = cell.get("latency_per_waveform_ms")
            per_waveform = (
                _normalized_summary(per_waveform_obj, unit="ms/waveform")
                if isinstance(per_waveform_obj, Mapping)
                else {
                    **latency,
                    "unit": "ms/waveform",
                    "samples": [value / batch for value in latency["samples"]],
                    "p50": latency["p50"] / batch,
                    "p95": (
                        latency["p95"] / batch
                        if latency["p95"] is not None
                        else None
                    ),
                    "p99": (
                        latency["p99"] / batch
                        if latency["p99"] is not None
                        else None
                    ),
                    "median_ci95": {
                        key: value / batch
                        for key, value in latency["median_ci95"].items()
                    },
                }
            )
            speedup_obj = cell.get("speedup_vs_reference")
            speedup = (
                _normalized_summary(
                    speedup_obj, unit=f"ratio-vs-{reference_route}"
                )
                if isinstance(speedup_obj, Mapping)
                else None
            )
            cold_obj = cell.get("cold_latency_block_ms")
            cold_latency = (
                _normalized_summary(cold_obj, unit="ms/block")
                if isinstance(cold_obj, Mapping)
                else None
            )
            memory_obj = cell.get("peak_cuda_memory")
            if isinstance(memory_obj, Mapping):
                allocated_obj = memory_obj.get("allocated_bytes")
                reserved_obj = memory_obj.get("reserved_bytes")
                route_data["peak_cuda_memory_measurement"] = {
                    key: memory_obj.get(key) for key in ("api", "boundary")
                }
            else:
                allocated_obj = reserved_obj = None
            allocated = (
                _normalized_summary(allocated_obj, unit="bytes")
                if isinstance(allocated_obj, Mapping)
                else None
            )
            reserved = (
                _normalized_summary(reserved_obj, unit="bytes")
                if isinstance(reserved_obj, Mapping)
                else None
            )
            route_data["latency_block_ms"].append(latency)
            route_data["cold_latency_block_ms"].append(cold_latency)
            route_data["latency_per_waveform_ms"].append(per_waveform)
            route_data["throughput_wps"].append(throughput)
            route_data["speedup_vs_reference"].append(speedup)
            route_data["peak_cuda_allocated_bytes"].append(allocated)
            route_data["peak_cuda_reserved_bytes"].append(reserved)
        normalized_routes[route] = route_data

    reference_throughput = normalized_routes[reference_route]["throughput_wps"]
    for route_data in normalized_routes.values():
        for index, speedup in enumerate(route_data["speedup_vs_reference"]):
            if speedup is None:
                route_data["speedup_vs_reference"][index] = _ratio_summary(
                    route_data["throughput_wps"][index],
                    reference_throughput[index],
                    unit=f"ratio-vs-{reference_route}",
                )

    return {
        "artifact_kind": "production_live_batch",
        "batches": batches,
        "routes": normalized_routes,
        "reference_route": reference_route,
        "measurement": dict(measurement),
        "metadata": data.get("metadata", data.get("runtime", {})),
        "workload": data.get("workload", {}),
        "parity_analysis": data.get("parity_analysis"),
    }


def _find_alias(cell: Mapping, aliases: Sequence[str]) -> Mapping | None:
    for alias in aliases:
        if alias in cell and cell[alias]:
            return cell[alias]
    return None


def _parse_component_suite(data: Mapping) -> dict:
    results = data.get("results", {})
    live = results.get("synthetic_live_batch")
    if not isinstance(live, Mapping) or not live:
        raise ValueError("component artifact has no synthetic_live_batch results")
    batches = sorted(int(key.removeprefix("batch_")) for key in live)
    available = []
    for route, aliases in ROUTE_ALIASES.items():
        if all(_find_alias(live[f"batch_{batch}"], aliases) for batch in batches):
            available.append(route)
    if not available:
        raise ValueError("component artifact has no complete live-batch route")
    reference_route = (
        "original_standard" if "original_standard" in available else available[0]
    )
    normalized_routes = {}
    for route in available:
        definition = _legacy_route_definition(route)
        route_data = {
            "definition": definition,
            "label": _route_label(route, definition),
            "latency_block_ms": [],
            "cold_latency_block_ms": [None] * len(batches),
            "latency_per_waveform_ms": [],
            "throughput_wps": [],
            "speedup_vs_reference": [],
            "peak_cuda_allocated_bytes": [None] * len(batches),
            "peak_cuda_reserved_bytes": [None] * len(batches),
            "peak_cuda_memory_measurement": None,
        }
        for batch in batches:
            raw = _find_alias(live[f"batch_{batch}"], ROUTE_ALIASES[route])
            latency = _normalized_summary(raw, scale=1000.0, unit="ms/block")
            per_waveform = _normalized_summary(
                raw, scale=1000.0 / batch, unit="ms/waveform"
            )
            throughput = _inverse_summary(
                raw, float(batch), unit="waveforms/second"
            )
            route_data["latency_block_ms"].append(latency)
            route_data["latency_per_waveform_ms"].append(per_waveform)
            route_data["throughput_wps"].append(throughput)
        normalized_routes[route] = route_data
    reference_throughput = normalized_routes[reference_route]["throughput_wps"]
    for route_data in normalized_routes.values():
        route_data["speedup_vs_reference"] = [
            _ratio_summary(
                throughput,
                reference_throughput[index],
                unit=f"ratio-vs-{reference_route}",
            )
            for index, throughput in enumerate(route_data["throughput_wps"])
        ]
    return {
        "artifact_kind": "component_microbenchmarks",
        "batches": batches,
        "routes": normalized_routes,
        "reference_route": reference_route,
        "measurement": {
            "scope": "component_internal",
            "call_surface": "synthetic LiveBatchMatchedFilter._process_batch harness",
            "claim_level": "private_component",
            "full_cli_end_to_end": {"measured": False},
            "limitations": [
                {"code": "not_production_or_full_cli_end_to_end"},
                {"code": "legacy_route_feature_flags_unknown"},
            ],
        },
        "metadata": data.get("metadata", {}),
        "workload": data.get("workload", {}),
        "parity_analysis": None,
    }


def parse_live_batch_artifact(data: Mapping) -> dict:
    """Normalize supported live-batch artifact generations for plotting."""

    if not isinstance(data, Mapping):
        raise TypeError("benchmark artifact must be a mapping")
    if isinstance(data.get("summary_by_batch"), Mapping):
        return _parse_production_live_batch(data)
    return _parse_component_suite(data)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def find_live_batch_artifact(directory: Path) -> tuple[Path, Mapping, dict]:
    """Choose public production evidence before component-only fallbacks."""

    parsed = []
    errors = []
    for name in LIVE_ARTIFACT_NAMES:
        path = directory / name
        if not path.is_file():
            continue
        try:
            data = _load_json(path)
            parsed.append((path, data, parse_live_batch_artifact(data)))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    for candidate in parsed:
        if candidate[2]["measurement"].get("scope") == "public_library_api":
            return candidate
    if parsed:
        return parsed[0]
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise ValueError(f"no usable live-batch artifact in {directory}{detail}")


def _measurement_title(evidence: Mapping) -> str:
    measurement = evidence["measurement"]
    scope = {
        "public_library_api": "Public library API",
        "component_internal": "Component-level only",
    }.get(measurement.get("scope"), str(measurement.get("scope", "Unknown scope")))
    return f"{scope}: {measurement.get('call_surface', 'unspecified call surface')}"


def _yerr(summaries: Sequence[Mapping]):
    import numpy as np

    centers = np.asarray([summary["p50"] for summary in summaries], dtype=float)
    lows = np.asarray(
        [summary["median_ci95"]["low"] for summary in summaries], dtype=float
    )
    highs = np.asarray(
        [summary["median_ci95"]["high"] for summary in summaries], dtype=float
    )
    return centers, np.vstack((centers - lows, highs - centers))


def _style_route(route: str, index: int):
    fallback = (f"C{index % 10}", "o", "-")
    return ROUTE_STYLE.get(route, fallback)


def _errorbar_route(ax, x, summaries, route, label, index, **kwargs):
    color, marker, linestyle = _style_route(route, index)
    points, errors = _yerr(summaries)
    return ax.errorbar(
        x,
        points,
        yerr=errors,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.8,
        markersize=6,
        capsize=3,
        label=label,
        **kwargs,
    )


def _configure_batch_axis(ax, batches, ylabel, title, *, log_y=True):
    import matplotlib.pyplot as plt

    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(batches)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Batch / bank templates")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.45)


def _save_figure(fig, output_dir: Path, filename: str, manifest: list[dict], entry: dict):
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    manifest.append({"path": str(path), **entry})
    print(f"Saved: {path}")


def plot_live_batch(evidence: Mapping, output_dir: Path, manifest: list[dict]):
    import matplotlib.pyplot as plt
    import numpy as np

    batches = np.asarray(evidence["batches"], dtype=int)
    routes = evidence["routes"]
    title = _measurement_title(evidence)
    common = {
        "source_kind": evidence["artifact_kind"],
        "evidence_scope": evidence["measurement"].get("scope"),
        "call_surface": evidence["measurement"].get("call_surface"),
        "full_cli_end_to_end": False,
    }

    fig, (throughput_ax, speedup_ax) = plt.subplots(1, 2, figsize=(14, 5.6))
    for index, (route, route_data) in enumerate(routes.items()):
        _errorbar_route(
            throughput_ax,
            batches,
            route_data["throughput_wps"],
            route,
            route_data["label"],
            index,
        )
        _errorbar_route(
            speedup_ax,
            batches,
            route_data["speedup_vs_reference"],
            route,
            route_data["label"],
            index,
        )
    _configure_batch_axis(
        throughput_ax,
        batches,
        "Throughput (waveforms / second; p50 ± median 95% CI)",
        "Throughput / batch-size tradeoff",
    )
    _configure_batch_axis(
        speedup_ax,
        batches,
        "Speedup ratio (p50 ± median 95% CI)",
        f"Speedup vs {evidence['reference_route']}",
        log_y=False,
    )
    speedup_ax.axhline(1.0, color="#4A5568", linewidth=1.0, linestyle=":")
    for ax in (throughput_ax, speedup_ax):
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "torch_live_batch_scaling.png",
        manifest,
        {**common, "metrics": ["throughput_wps", "speedup_vs_reference"]},
    )
    plt.close(fig)

    fig, (latency_ax, tail_ax) = plt.subplots(1, 2, figsize=(14, 5.6))
    for index, (route, route_data) in enumerate(routes.items()):
        color, marker, _linestyle = _style_route(route, index)
        _errorbar_route(
            latency_ax,
            batches,
            route_data["latency_block_ms"],
            route,
            f"{route_data['label']} p50",
            index,
        )
        p95 = [summary["p95"] for summary in route_data["latency_block_ms"]]
        p99 = [summary["p99"] for summary in route_data["latency_block_ms"]]
        if all(value is not None for value in p95):
            tail_ax.plot(
                batches,
                p95,
                color=color,
                marker=marker,
                linestyle="--",
                label=f"{route_data['label']} p95",
            )
        if all(value is not None for value in p99):
            tail_ax.plot(
                batches,
                p99,
                color=color,
                marker=marker,
                linestyle=":",
                label=f"{route_data['label']} p99",
            )
    _configure_batch_axis(
        latency_ax,
        batches,
        "Latency per block (ms; p50 ± median 95% CI)",
        "Median latency",
    )
    _configure_batch_axis(
        tail_ax,
        batches,
        "Latency per block (ms)",
        "Tail latency from retained artifact samples",
    )
    latency_ax.legend(frameon=True, fontsize=8)
    if tail_ax.lines:
        tail_ax.legend(frameon=True, fontsize=7)
    else:
        tail_ax.text(
            0.5,
            0.5,
            "p95/p99 unavailable in this legacy artifact",
            transform=tail_ax.transAxes,
            ha="center",
            va="center",
        )
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "torch_latency_breakdown.png",
        manifest,
        {**common, "metrics": ["latency_block_ms.p50", "p95", "p99"]},
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for index, (route, route_data) in enumerate(routes.items()):
        _errorbar_route(
            axes[0, 0], batches, route_data["throughput_wps"], route,
            route_data["label"], index
        )
        _errorbar_route(
            axes[0, 1], batches, route_data["speedup_vs_reference"], route,
            route_data["label"], index
        )
        _errorbar_route(
            axes[1, 0], batches, route_data["latency_block_ms"], route,
            route_data["label"], index
        )
        _errorbar_route(
            axes[1, 1], batches, route_data["latency_per_waveform_ms"], route,
            route_data["label"], index
        )
    _configure_batch_axis(axes[0, 0], batches, "Waveforms / second", "Throughput")
    _configure_batch_axis(
        axes[0, 1], batches, "Speedup ratio", "Speedup with uncertainty", log_y=False
    )
    axes[0, 1].axhline(1.0, color="#4A5568", linewidth=1.0, linestyle=":")
    _configure_batch_axis(
        axes[1, 0], batches, "ms / block", "p50 block latency"
    )
    _configure_batch_axis(
        axes[1, 1], batches, "ms / waveform", "Amortized p50 latency"
    )
    for ax in axes.flat:
        ax.legend(frameon=True, fontsize=7)
    fig.suptitle(f"PyCBC Torch performance evidence\n{title}")
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "torch_performance_dashboard.png",
        manifest,
        {
            **common,
            "metrics": [
                "throughput_wps",
                "speedup_vs_reference",
                "latency_block_ms",
                "latency_per_waveform_ms",
            ],
        },
    )
    plt.close(fig)


def plot_cold_warm(
    evidence: Mapping,
    output_dir: Path,
    manifest: list[dict],
):
    """Render retained first-use and steady-state latency separately."""

    import matplotlib.pyplot as plt
    import numpy as np

    batches = np.asarray(evidence["batches"], dtype=int)
    available = {
        route: route_data
        for route, route_data in evidence["routes"].items()
        if all(
            summary is not None
            for summary in route_data["cold_latency_block_ms"]
        )
    }
    if not available:
        return

    fig, (latency_ax, ratio_ax) = plt.subplots(1, 2, figsize=(14, 5.6))
    for index, (route, route_data) in enumerate(available.items()):
        color, marker, _linestyle = _style_route(route, index)
        warm = route_data["latency_block_ms"]
        cold = route_data["cold_latency_block_ms"]
        warm_points, warm_errors = _yerr(warm)
        cold_points, cold_errors = _yerr(cold)
        latency_ax.errorbar(
            batches,
            cold_points,
            yerr=cold_errors,
            color=color,
            marker=marker,
            linestyle="-",
            capsize=3,
            label=f"{route_data['label']} cold",
        )
        latency_ax.errorbar(
            batches,
            warm_points,
            yerr=warm_errors,
            color=color,
            marker=marker,
            markerfacecolor="white",
            linestyle="--",
            capsize=3,
            label=f"{route_data['label']} warm",
        )
        ratios = [
            _ratio_summary(
                cold_summary,
                warm[index],
                unit="cold/warm latency ratio",
            )
            for index, cold_summary in enumerate(cold)
        ]
        ratio_points, ratio_errors = _yerr(ratios)
        ratio_ax.errorbar(
            batches,
            ratio_points,
            yerr=ratio_errors,
            color=color,
            marker=marker,
            capsize=3,
            label=route_data["label"],
        )

    _configure_batch_axis(
        latency_ax,
        batches,
        "Latency per block (ms; p50 ± median 95% CI)",
        "Cold and warm samples kept separate",
    )
    _configure_batch_axis(
        ratio_ax,
        batches,
        "Cold / warm p50 latency ratio",
        "First-use overhead",
        log_y=False,
    )
    ratio_ax.axhline(1.0, color="#4A5568", linewidth=1.0, linestyle=":")
    latency_ax.legend(frameon=True, fontsize=7)
    ratio_ax.legend(frameon=True, fontsize=7)
    fig.suptitle(_measurement_title(evidence))
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "torch_cold_warm_latency.png",
        manifest,
        {
            "source_kind": evidence["artifact_kind"],
            "evidence_scope": evidence["measurement"].get("scope"),
            "call_surface": evidence["measurement"].get("call_surface"),
            "full_cli_end_to_end": False,
            "metrics": [
                "cold_latency_block_ms",
                "latency_block_ms",
                "cold_to_warm_latency_ratio",
            ],
        },
    )
    plt.close(fig)


def plot_cuda_memory(
    evidence: Mapping,
    output_dir: Path,
    manifest: list[dict],
):
    """Render CUDA allocator peaks when the artifact retained both APIs."""

    import matplotlib.pyplot as plt

    batches = evidence["batches"]
    available = []
    for route, route_data in evidence["routes"].items():
        cells = [
            (batch, allocated, reserved)
            for batch, allocated, reserved in zip(
                batches,
                route_data["peak_cuda_allocated_bytes"],
                route_data["peak_cuda_reserved_bytes"],
                strict=True,
            )
            if allocated is not None and reserved is not None
        ]
        if cells:
            available.append((route, route_data, cells))
    if not available:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    measurement = None
    for index, (route, route_data, cells) in enumerate(available):
        color, marker, _linestyle = _style_route(route, index)
        route_batches = [cell[0] for cell in cells]
        allocated = [
            _normalized_summary(
                cell[1], scale=1.0 / (1024.0**2), unit="MiB"
            )
            for cell in cells
        ]
        reserved = [
            _normalized_summary(
                cell[2], scale=1.0 / (1024.0**2), unit="MiB"
            )
            for cell in cells
        ]
        allocated_points, allocated_errors = _yerr(allocated)
        reserved_points, reserved_errors = _yerr(reserved)
        ax.errorbar(
            route_batches,
            allocated_points,
            yerr=allocated_errors,
            color=color,
            marker=marker,
            linestyle="-",
            capsize=3,
            label=f"{route_data['label']} allocated",
        )
        ax.errorbar(
            route_batches,
            reserved_points,
            yerr=reserved_errors,
            color=color,
            marker=marker,
            markerfacecolor="white",
            linestyle="--",
            capsize=3,
            label=f"{route_data['label']} reserved",
        )
        measurement = route_data["peak_cuda_memory_measurement"]

    _configure_batch_axis(
        ax,
        batches,
        "Peak CUDA memory (MiB; p50 ± median 95% CI)",
        "Torch CUDA allocator peaks",
    )
    ax.legend(frameon=True, fontsize=7)
    fig.suptitle(_measurement_title(evidence))
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "torch_cuda_memory.png",
        manifest,
        {
            "source_kind": evidence["artifact_kind"],
            "evidence_scope": evidence["measurement"].get("scope"),
            "call_surface": evidence["measurement"].get("call_surface"),
            "full_cli_end_to_end": False,
            "metrics": [
                "peak_cuda_memory.allocated_bytes",
                "peak_cuda_memory.reserved_bytes",
            ],
            "measurement": measurement,
            "oom_cells_included": False,
        },
    )
    plt.close(fig)


def _parity_cells(evidence: Mapping) -> dict[str, dict[int, dict]]:
    analysis = evidence.get("parity_analysis")
    if not isinstance(analysis, Mapping):
        return {}
    cells: dict[str, dict[int, dict]] = {}
    metrics = (
        "max_snr_diff",
        "max_phase_diff",
        "max_sigmasq_relative_diff",
        "relative_output_l2_diff",
    )
    for batch in evidence["batches"]:
        batch_result = analysis.get(f"batch_{batch}")
        if not isinstance(batch_result, Mapping):
            continue
        replicates = batch_result.get("replicates")
        if not isinstance(replicates, Mapping):
            replicates = {"replicate_0": batch_result}
        by_comparison: dict[str, list[Mapping]] = {}
        for replicate in replicates.values():
            comparisons = (
                replicate.get("comparisons", {})
                if isinstance(replicate, Mapping)
                else {}
            )
            for comparison, result in comparisons.items():
                if isinstance(result, Mapping):
                    by_comparison.setdefault(comparison, []).append(result)
        for comparison, results in by_comparison.items():
            cells.setdefault(comparison, {})[int(batch)] = {
                "passed": all(bool(result.get("passed")) for result in results),
                "replicate_count": len(results),
                **{
                    metric: max(float(result.get(metric, 0.0)) for result in results)
                    for metric in metrics
                },
            }
    return cells


def plot_parity(
    evidence: Mapping,
    output_dir: Path,
    manifest: list[dict],
):
    """Plot numerical maxima and pass/fail state for every retained replicate."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    cells = _parity_cells(evidence)
    if not cells:
        return
    batches = list(evidence["batches"])
    metrics = (
        ("max_snr_diff", "Maximum SNR difference"),
        ("max_phase_diff", "Maximum phase difference"),
        ("max_sigmasq_relative_diff", "Maximum relative sigmasq difference"),
        ("relative_output_l2_diff", "Relative output L2 difference"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for metric_index, (metric, title) in enumerate(metrics):
        ax = axes.flat[metric_index]
        for comparison_index, (comparison, comparison_cells) in enumerate(
            cells.items()
        ):
            route_batches = [
                batch for batch in batches if batch in comparison_cells
            ]
            values = [comparison_cells[batch][metric] for batch in route_batches]
            ax.plot(
                route_batches,
                values,
                color=f"C{comparison_index % 10}",
                marker="o",
                label=comparison.replace("_vs_", " vs "),
            )
        ax.axhline(1.0e-3, color="#C53030", linestyle=":", label="1e-3 limit")
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=1.0e-12)
        ax.set_xticks(batches)
        ax.set_xticklabels([str(batch) for batch in batches])
        ax.set_xlabel("Batch / bank templates")
        ax.set_ylabel("Maximum across replicates")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.45)
        ax.legend(frameon=True, fontsize=6)

    comparisons = list(cells)
    pass_matrix = np.full((len(comparisons), len(batches)), np.nan)
    for row, comparison in enumerate(comparisons):
        for column, batch in enumerate(batches):
            if batch in cells[comparison]:
                pass_matrix[row, column] = float(
                    cells[comparison][batch]["passed"]
                )
    pass_ax = axes.flat[4]
    color_map = ListedColormap(["#C53030", "#38A169"])
    color_map.set_bad("#A0AEC0")
    image = pass_ax.imshow(
        np.ma.masked_invalid(pass_matrix),
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap=color_map,
    )
    pass_ax.set_xticks(range(len(batches)))
    pass_ax.set_xticklabels([str(batch) for batch in batches])
    pass_ax.set_yticks(range(len(comparisons)))
    pass_ax.set_yticklabels(
        [comparison.replace("_vs_", " vs ") for comparison in comparisons],
        fontsize=7,
    )
    pass_ax.set_xlabel("Batch / bank templates")
    pass_ax.set_title("All-replicate parity outcome")
    color_bar = fig.colorbar(image, ax=pass_ax, ticks=[0, 1], fraction=0.046)
    color_bar.ax.set_yticklabels(["fail", "pass"])

    note_ax = axes.flat[5]
    note_ax.axis("off")
    note_ax.text(
        0.02,
        0.95,
        "A cell passes only when every retained replicate passes:\n"
        "• continuous error metrics are below 1e-3\n"
        "• block/trigger counts and sequence lengths match\n"
        "• template IDs, end times, and veto counts match\n\n"
        "Gray means that comparison was not measured.",
        va="top",
        fontsize=10,
    )
    fig.suptitle(
        f"Scientific parity for timed routes\n{_measurement_title(evidence)}"
    )
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "torch_parity_error.png",
        manifest,
        {
            "source_kind": evidence["artifact_kind"],
            "evidence_scope": evidence["measurement"].get("scope"),
            "call_surface": evidence["measurement"].get("call_surface"),
            "full_cli_end_to_end": False,
            "metrics": [metric for metric, _title in metrics] + ["passed"],
            "aggregation": "maximum error and logical all across replicates",
            "continuous_tolerance": 1.0e-3,
        },
    )
    plt.close(fig)


def _find_optional(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def plot_inference_components(
    suite_data: Mapping,
    output_dir: Path,
    manifest: list[dict],
):
    import matplotlib.pyplot as plt

    results = suite_data.get("results", {})
    detector = results.get("detector_network")
    relbin = results.get("relbin_likelihood")
    if not isinstance(detector, Mapping) or not isinstance(relbin, Mapping):
        return
    detector_points = sorted(int(key.removeprefix("points_")) for key in detector)
    relbin_samples = sorted(int(key.removeprefix("samples_")) for key in relbin)
    fig, (detector_ax, relbin_ax) = plt.subplots(1, 2, figsize=(14, 5.6))

    detector_routes = (
        ("Sequential", ("sequential", "sequential_single"), "#4A5568", "d"),
        ("Vectorized", ("vectorized", "vectorized_network"), "#3182CE", "o"),
    )
    for label, aliases, color, marker in detector_routes:
        summaries = [
            _get_route(detector[f"points_{point}"], *aliases)
            for point in detector_points
        ]
        points, errors = _yerr(
            [_normalized_summary(item, scale=1000.0, unit="ms") for item in summaries]
        )
        detector_ax.errorbar(
            detector_points, points, yerr=errors, color=color, marker=marker,
            capsize=3, label=label
        )
    detector_ax.set_xscale("log")
    detector_ax.set_yscale("log")
    detector_ax.set_xlabel("Sky grid points")
    detector_ax.set_ylabel("Component latency (ms)")
    detector_ax.set_title("Detector-network component (p50 ± median 95% CI)")
    detector_ax.legend()
    detector_ax.grid(True, which="both", linestyle="--", alpha=0.45)

    relbin_routes = (
        ("Sequential", ("sequential", "sequential_single"), "#4A5568", "d"),
        ("Batched", ("batched", "batched_summary"), "#38A169", "s"),
    )
    for label, aliases, color, marker in relbin_routes:
        summaries = [
            _get_route(relbin[f"samples_{sample}"], *aliases)
            for sample in relbin_samples
        ]
        points, errors = _yerr(
            [_normalized_summary(item, scale=1000.0, unit="ms") for item in summaries]
        )
        relbin_ax.errorbar(
            relbin_samples, points, yerr=errors, color=color, marker=marker,
            capsize=3, label=label
        )
    relbin_ax.set_xscale("log")
    relbin_ax.set_yscale("log")
    relbin_ax.set_xlabel("Parameter samples")
    relbin_ax.set_ylabel("Component latency (ms)")
    relbin_ax.set_title("Relative-binning component (p50 ± median 95% CI)")
    relbin_ax.legend()
    relbin_ax.grid(True, which="both", linestyle="--", alpha=0.45)
    fig.suptitle("Component-level benchmarks; not full inference latency")
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "torch_inference_acceleration.png",
        manifest,
        {
            "source_kind": "component_microbenchmarks",
            "evidence_scope": "component_internal",
            "full_cli_end_to_end": False,
            "metrics": ["detector_network_latency", "relbin_summary_latency"],
        },
    )
    plt.close(fig)


def plot_matched_filter(
    data: Mapping,
    output_dir: Path,
    manifest: list[dict],
):
    import matplotlib.pyplot as plt

    campaigns = data.get("campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        return
    results = campaigns[0].get("results", {})
    size_keys = sorted(
        (key for key in results if key.startswith("size_")),
        key=lambda key: int(key.removeprefix("size_")),
    )
    if not size_keys:
        return
    sizes = [int(key.removeprefix("size_")) for key in size_keys]
    route_names = [
        route
        for route in (
            "standard_1t",
            "standard_16t",
            "torch_cpu_16t",
            "torch_cuda_eager",
            "torch_cuda",
        )
        if all(results[key].get(route) for key in size_keys)
    ]
    if not route_names:
        return
    route_colors = ("#4A5568", "#718096", "#3182CE", "#38A169", "#805AD5")
    fig, (latency_ax, speedup_ax) = plt.subplots(1, 2, figsize=(14, 5.6))
    normalized = {}
    for index, route in enumerate(route_names):
        summaries = [
            _normalized_summary(results[key][route], scale=1000.0, unit="ms")
            for key in size_keys
        ]
        normalized[route] = summaries
        points, errors = _yerr(summaries)
        latency_ax.errorbar(
            sizes, points, yerr=errors, color=route_colors[index], marker="o",
            capsize=3, label=route.replace("_", " ")
        )
    latency_ax.set_xscale("log", base=2)
    latency_ax.set_yscale("log")
    latency_ax.set_xlabel("FFT samples")
    latency_ax.set_ylabel("Latency (ms)")
    latency_ax.set_title("Public matched-filter method latency")
    latency_ax.legend(fontsize=8)
    latency_ax.grid(True, which="both", linestyle="--", alpha=0.45)

    reference_name = "standard_1t" if "standard_1t" in normalized else route_names[0]
    reference = normalized[reference_name]
    for index, route in enumerate(route_names):
        ratios = [
            _ratio_summary(reference[cell], summary, unit=f"ratio-vs-{reference_name}")
            for cell, summary in enumerate(normalized[route])
        ]
        points, errors = _yerr(ratios)
        speedup_ax.errorbar(
            sizes, points, yerr=errors, color=route_colors[index], marker="o",
            capsize=3, label=route.replace("_", " ")
        )
    speedup_ax.axhline(1.0, color="#4A5568", linewidth=1.0, linestyle=":")
    speedup_ax.set_xscale("log", base=2)
    speedup_ax.set_xlabel("FFT samples")
    speedup_ax.set_ylabel("Speedup ratio with median 95% CI")
    speedup_ax.set_title(f"Speedup vs {reference_name.replace('_', ' ')}")
    speedup_ax.legend(fontsize=8)
    speedup_ax.grid(True, which="both", linestyle="--", alpha=0.45)
    fig.suptitle(
        "Library-component evidence: "
        "MatchedFilterControl.full_matched_filter_and_cluster_symm"
    )
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "torch_matched_filter_symm_scaling.png",
        manifest,
        {
            "source_kind": "matched_filter_symm",
            "evidence_scope": "public_library_api",
            "call_surface": "MatchedFilterControl.full_matched_filter_and_cluster_symm",
            "full_cli_end_to_end": False,
            "metrics": ["latency_ms", "speedup_vs_standard_1t"],
        },
    )
    plt.close(fig)


def plot_cpu_profile(data, output_dir: Path, manifest: list[dict]):
    import matplotlib.pyplot as plt
    import numpy as np

    if not isinstance(data, list) or not data:
        return
    sizes = sorted({int(item["N"]) for item in data})
    threads = sorted({int(item["threads"]) for item in data})
    cells = {(int(item["N"]), int(item["threads"])): item for item in data}
    if any((size, thread) not in cells for size in sizes for thread in threads):
        return
    fig, (scaling_ax, components_ax) = plt.subplots(1, 2, figsize=(14, 5.6))
    for index, size in enumerate(sizes):
        summaries = [
            _normalized_summary(
                cells[(size, thread)]["full_us"], scale=0.001, unit="ms"
            )
            for thread in threads
        ]
        points, errors = _yerr(summaries)
        scaling_ax.errorbar(
            threads, points, yerr=errors, marker="o", capsize=3,
            color=f"C{index % 10}", label=f"N={size:,}"
        )
    scaling_ax.set_xscale("log", base=2)
    scaling_ax.set_yscale("log")
    scaling_ax.set_xticks(threads)
    scaling_ax.set_xticklabels([f"{thread}T" for thread in threads])
    scaling_ax.set_xlabel("CPU threads")
    scaling_ax.set_ylabel("Component latency (ms)")
    scaling_ax.set_title("CPU component scaling (p50 ± median 95% CI)")
    scaling_ax.legend(fontsize=8)
    scaling_ax.grid(True, which="both", linestyle="--", alpha=0.45)

    largest = sizes[-1]
    component_names = ("corr_us", "ifft_us", "cluster_us")
    labels = ("Correlation", "IFFT", "Clustering")
    colors = ("#63B3ED", "#3182CE", "#DD6B20")
    bottom = np.zeros(len(threads))
    positions = np.arange(len(threads))
    for name, label, color in zip(component_names, labels, colors, strict=True):
        values = np.asarray(
            [
                summary_statistic(cells[(largest, thread)][name], "p50") / 1000.0
                for thread in threads
            ]
        )
        components_ax.bar(
            positions, values, bottom=bottom, color=color, label=label,
            edgecolor="black", linewidth=0.4
        )
        bottom += values
    components_ax.set_xticks(positions)
    components_ax.set_xticklabels([f"{thread}T" for thread in threads])
    components_ax.set_xlabel("CPU threads")
    components_ax.set_ylabel("Median component time (ms)")
    components_ax.set_title(f"Measured component breakdown at N={largest:,}")
    components_ax.legend(fontsize=8)
    components_ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    fig.suptitle("Private component profile; not end-to-end search latency")
    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "torch_cpu_thread_scaling.png",
        manifest,
        {
            "source_kind": "cpu_component_profile",
            "evidence_scope": "component_internal",
            "full_cli_end_to_end": False,
            "metrics": ["full_us", *component_names],
        },
    )
    plt.close(fig)


def _configure_plot_style():
    import matplotlib.pyplot as plt

    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.style.use(style)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
        }
    )


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate scoped PyCBC Torch performance-evidence plots."
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        type=Path,
        help="Directory containing measured benchmark JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "docs" / "images",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifacts_dir = args.artifacts_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    live_path, live_data, evidence = find_live_batch_artifact(artifacts_dir)
    print(f"Loaded live-batch evidence from {live_path}")
    generated: list[dict] = []
    plot_live_batch(evidence, output_dir, generated)
    plot_cold_warm(evidence, output_dir, generated)
    plot_cuda_memory(evidence, output_dir, generated)
    plot_parity(evidence, output_dir, generated)

    suite_data = None
    if "results" in live_data:
        suite_data = live_data
    else:
        suite_path = _find_optional(
            artifacts_dir,
            (
                "extended_full_suite.json",
                "comprehensive_benchmark_results.json",
                "len_extended_full_suite.json",
            ),
        )
        if suite_path is not None:
            suite_data = _load_json(suite_path)
    if suite_data is not None:
        plot_inference_components(suite_data, output_dir, generated)

    matched_filter_path = _find_optional(
        artifacts_dir, MATCHED_FILTER_ARTIFACT_NAMES
    )
    if matched_filter_path is not None:
        plot_matched_filter(
            _load_json(matched_filter_path), output_dir, generated
        )

    cpu_profile_path = _find_optional(
        artifacts_dir, ("cpu_profile_breakdown.json",)
    )
    if cpu_profile_path is not None:
        plot_cpu_profile(
            _load_json(cpu_profile_path), output_dir, generated
        )

    measurement = evidence["measurement"]
    manifest = {
        "schema_version": 1,
        "artifact_type": "torch_performance_plot_manifest",
        "source_artifact": str(live_path),
        "measurement": measurement,
        "source_metadata": evidence.get("metadata", {}),
        "workload": evidence.get("workload", {}),
        "routes": {
            route: route_data["definition"]
            for route, route_data in evidence["routes"].items()
        },
        "dimensions": {
            "batch_templates": evidence["batches"],
            "latency_statistics": ["p50", "p95", "p99"],
            "uncertainty": {
                "statistic": "median",
                "confidence": 0.95,
                "source": "artifact median_ci95 or retained samples",
                "missing_policy": "point estimate only; interval marked unavailable",
            },
        },
        "generated_plots": generated,
        "full_cli_end_to_end_claim_supported": bool(
            measurement.get("full_cli_end_to_end", {}).get("measured", False)
        ),
    }
    manifest_path = output_dir / "torch_performance_plot_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(f"Saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
