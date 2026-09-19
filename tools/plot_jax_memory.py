#!/usr/bin/env python3
"""Plot the JSON receipt written by :mod:`profile_jax_memory`.

The live-array inventory can contain several views of one device buffer.  The
primary series therefore groups by ``(device, unsafe_buffer_pointer)`` and
uses the largest ``nbytes`` value in each group.  Records without a usable
pointer are reported as unavailable instead of being treated as zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_INPUT = Path(
    "artifacts/jax-nsight-memory-20260919/snapshots/jax_memory_profile.json"
)
DEFAULT_OUTPUT = Path("docs/_static/jax_memory_profile.png")


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def distinct_live_bytes(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Return de-duplicated live-array bytes and pointer availability.

    A pointer key contributes the maximum ``nbytes`` among its records, which
    prevents views/aliases from inflating the distinct-buffer series.  Missing
    pointers remain outside that sum and are explicitly counted as unknown.
    """
    inventory = snapshot.get("live_arrays")
    if not isinstance(inventory, dict) or "arrays" not in inventory:
        return {
            "distinct_bytes": None,
            "unknown_count": None,
            "unknown_logical_bytes": None,
        }
    groups: Dict[Tuple[str, Any], float] = {}
    unknown_count = 0
    unknown_logical_bytes = 0.0
    for record in inventory.get("arrays") or []:
        if not isinstance(record, dict):
            unknown_count += 1
            continue
        nbytes = _number(record.get("nbytes"))
        pointer = record.get("unsafe_buffer_pointer")
        device = record.get("device")
        if pointer is None or device is None or nbytes is None:
            unknown_count += 1
            if nbytes is not None:
                unknown_logical_bytes += max(0.0, nbytes)
            continue
        key = (str(device), pointer)
        groups[key] = max(groups.get(key, 0.0), max(0.0, nbytes))
    return {
        "distinct_bytes": sum(groups.values()),
        "unknown_count": unknown_count,
        "unknown_logical_bytes": unknown_logical_bytes,
    }


def _logical_live_bytes(snapshot: Dict[str, Any]) -> Optional[float]:
    inventory = snapshot.get("live_arrays")
    if not isinstance(inventory, dict) or "arrays" not in inventory:
        return None
    values = [
        _number(record.get("nbytes"))
        for record in inventory.get("arrays") or []
        if isinstance(record, dict)
    ]
    return sum(max(0.0, value) for value in values if value is not None)


def _allocator_bytes(
    snapshot: Dict[str, Any], field: str
) -> Optional[float]:
    stats = snapshot.get("device_memory_stats")
    if not isinstance(stats, list):
        return None
    values = []
    for device in stats:
        if not isinstance(device, dict):
            continue
        memory_stats = device.get("memory_stats")
        if not isinstance(memory_stats, dict):
            continue
        value = _number(memory_stats.get(field))
        if value is not None:
            values.append(max(0.0, value))
    return sum(values) if values else None


def _series(snapshots: Iterable[Dict[str, Any]], field: str) -> np.ndarray:
    values = []
    for snapshot in snapshots:
        value = _allocator_bytes(snapshot, field)
        values.append(np.nan if value is None else value / 2**20)
    return np.asarray(values, dtype=float)


def plot_memory_profile(
    data: Dict[str, Any], output_path: Path, show_logical_sum: bool = False
) -> None:
    """Render device allocator/live-buffer and host RSS snapshots."""
    snapshots = data.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("Memory receipt has no snapshots")
    snapshots = [item for item in snapshots if isinstance(item, dict)]
    if not snapshots:
        raise ValueError("Memory receipt has no snapshot objects")

    x = np.arange(len(snapshots), dtype=float)
    labels = []
    for i, item in enumerate(snapshots):
        label = str(item.get("label", f"snapshot {i + 1}"))
        parts = label.split("_")
        if len(parts) >= 3 and parts[:2] == ["production", "batch"]:
            label = f"B{parts[2]}"
        elif label == "postcleanup":
            label = "After cleanup"
        labels.append(label)
    distinct = [distinct_live_bytes(item) for item in snapshots]
    distinct_mib = np.asarray([
        np.nan if item["distinct_bytes"] is None
        else item["distinct_bytes"] / 2**20
        for item in distinct
    ])
    logical_mib = np.asarray([
        np.nan if (value := _logical_live_bytes(item)) is None
        else value / 2**20
        for item in snapshots
    ])
    rss_mib = np.asarray([
        np.nan if (value := _number(item.get("process_rss_bytes"))) is None
        else value / 2**20
        for item in snapshots
    ])

    fig, (device_ax, host_ax) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": (1.35, 0.8), "hspace": 0.16},
    )
    device_lines = [
        (distinct_mib, "Distinct live buffers (known pointers)", "#218c4c"),
        (
            _series(snapshots, "bytes_in_use"),
            "Allocator bytes in use", "#0072b2",
        ),
        (_series(snapshots, "pool_bytes"), "Allocator pool", "#d55e00"),
        (
            _series(snapshots, "peak_bytes_in_use"),
            "Peak bytes in use", "#7b2cbf",
        ),
    ]
    if show_logical_sum:
        device_lines.append((
            logical_mib,
            "Logical live-array sum (aliases duplicated)",
            "#666666",
        ))
    for values, label, color in device_lines:
        device_ax.plot(x, values, marker="o", lw=1.8, ms=4, label=label,
                       color=color)
    device_ax.set_ylabel("Device memory (MiB)")
    device_ax.set_title(
        "JAX device memory snapshots", loc="left", fontweight="bold"
    )
    device_ax.grid(axis="y", linestyle=":", alpha=0.4)
    device_ax.legend(loc="best", fontsize=8, framealpha=0.9)

    host_ax.plot(x, rss_mib, marker="o", lw=1.8, ms=4,
                 color="#555555", label="Host RSS (sampled)")
    host_ax.set_ylabel("Host RSS (MiB)")
    host_ax.set_xlabel("Snapshot")
    finite_rss = rss_mib[np.isfinite(rss_mib)]
    if len(finite_rss):
        host_ax.set_ylim(top=max(1.0, float(finite_rss.max()) * 1.1))
    host_ax.grid(axis="y", linestyle=":", alpha=0.4)

    unknown = [item["unknown_count"] for item in distinct]
    unknown_labels = [
        f"? {count} pointer{'' if count == 1 else 's'} unavailable"
        for count in unknown if count not in (None, 0)
    ]
    if unknown_labels:
        device_ax.text(
            0.01, 0.02,
            "Unknown-pointer live arrays are unavailable, not zero: "
            + ", ".join(unknown_labels),
            transform=device_ax.transAxes, fontsize=8, color="#8a4b08",
        )
    host_ax.set_xticks(x, labels, rotation=35, ha="right")
    for axis in (device_ax, host_ax):
        axis.set_ylim(bottom=0)
        for spine in axis.spines.values():
            spine.set_color("#b0b7bf")
    fig.suptitle(
        f"JAX memory receipt · {len(snapshots)} snapshots", fontsize=13,
        fontweight="bold", y=0.98,
    )
    fig.text(
        0.01, 0.01,
        "Live buffers exclude JIT temporaries · allocator pool is not all "
        "live buffers",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.18)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--show-logical-sum", action="store_true",
        help="show the raw live-array sum, including aliases",
    )
    args = parser.parse_args(argv)
    with args.input.open(encoding="utf-8") as stream:
        data = json.load(stream)
    plot_memory_profile(data, args.output, args.show_logical_sum)
    print(
        f"[INFO] Successfully generated memory profiling plot: {args.output}"
    )


if __name__ == "__main__":
    main()
