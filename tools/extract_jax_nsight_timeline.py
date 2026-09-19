#!/usr/bin/env python3
"""Extract process-scoped CUDA activity from an Nsight Systems SQLite export.

The output is the input telemetry receipt with a ``cuda_trace`` member added.
CUDA times are expressed in seconds relative to the unique NVTX mark
``pycbc.timeline.origin``.  Transfer bytes are charged to the bin containing
the transfer completion timestamp; they are event counts, not a bandwidth
estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


ORIGIN_MARK = "pycbc.timeline.origin"
COPY_DIRECTIONS = {1: "H2D", 2: "D2H", 8: "D2D"}


class NsightTimelineError(ValueError):
    """Raised when a trace cannot be interpreted without guessing."""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute("PRAGMA table_info(\"%s\")" % table).fetchall()
    return {str(row[1]) for row in rows}


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _find_origin_ns(conn: sqlite3.Connection) -> int:
    if not _has_table(conn, "NVTX_EVENTS"):
        raise NsightTimelineError("Nsight export has no NVTX_EVENTS table")
    columns = _table_columns(conn, "NVTX_EVENTS")
    clauses = []
    params: List[Any] = []
    if "text" in columns:
        clauses.append("text = ?")
        params.append(ORIGIN_MARK)
    if "textId" in columns and _has_table(conn, "StringIds"):
        clauses.append(
            "textId IN (SELECT id FROM StringIds WHERE value = ?)"
        )
        params.append(ORIGIN_MARK)
    if not clauses:
        raise NsightTimelineError(
            "Nsight export has no NVTX text or StringIds columns"
        )
    rows = conn.execute(
        "SELECT start FROM NVTX_EVENTS WHERE " + " OR ".join(clauses), params
    ).fetchall()
    if len(rows) != 1:
        raise NsightTimelineError(
            "expected exactly one %r NVTX marker, found %d"
            % (ORIGIN_MARK, len(rows))
        )
    if rows[0][0] is None:
        raise NsightTimelineError("NVTX origin marker has no timestamp")
    return int(rows[0][0])


def _resolve_global_pid(conn: sqlite3.Connection, target_pid: int) -> int:
    """Resolve a Linux PID to Nsight's serialized process GlobalId.

    Recent exports use ``PROCESSES`` as the authoritative mapping.  Some
    synthetic exports and older Nsight versions expose the conventional
    ``pid << 24`` representation, which is retained as a strict fallback.
    """
    if target_pid <= 0:
        raise NsightTimelineError("target_pid must be a positive integer")
    candidates: set[int] = set()
    if _has_table(conn, "PROCESSES"):
        rows = conn.execute(
            "SELECT DISTINCT globalPid FROM PROCESSES "
            "WHERE pid = ? AND globalPid IS NOT NULL",
            (target_pid,),
        ).fetchall()
        candidates.update(int(row[0]) for row in rows)
    if len(candidates) > 1:
        raise NsightTimelineError(
            "target_pid %d maps to multiple Nsight globalPid values: %s"
            % (target_pid, sorted(candidates))
        )
    if candidates:
        return candidates.pop()

    conventional = target_pid << 24
    event_candidates: set[int] = set()
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if (
            _has_table(conn, table)
            and "globalPid" in _table_columns(conn, table)
        ):
            rows = conn.execute(
                "SELECT DISTINCT globalPid FROM \"%s\" "
                "WHERE globalPid IS NOT NULL" % table
            ).fetchall()
            event_candidates.update(int(row[0]) for row in rows)
    if conventional in event_candidates:
        return conventional
    raise NsightTimelineError(
        "no Nsight globalPid found for target_pid %d" % target_pid
    )


def _memory_kind(conn: sqlite3.Connection, value: Any) -> str:
    if value is None:
        return "Unknown"
    value = int(value)
    if _has_table(conn, "ENUM_CUDA_MEM_KIND"):
        row = conn.execute(
            "SELECT label, name FROM ENUM_CUDA_MEM_KIND WHERE id = ?", (value,)
        ).fetchone()
        if row:
            return str(row[0] or row[1] or "Unknown")
    return "Unknown (%d)" % value


def _target_devices(
    conn: sqlite3.Connection, global_pid: int, device_id: Optional[int]
) -> int:
    devices: set[int] = set()
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if not _has_table(conn, table):
            continue
        columns = _table_columns(conn, table)
        if "globalPid" not in columns or "deviceId" not in columns:
            continue
        rows = conn.execute(
            "SELECT DISTINCT deviceId FROM \"%s\" "
            "WHERE globalPid = ? AND deviceId IS NOT NULL" % table,
            (global_pid,),
        ).fetchall()
        devices.update(int(row[0]) for row in rows)
    if _has_table(conn, "TARGET_INFO_CUDA_DEVICE") and _has_table(
        conn, "PROCESSES"
    ):
        rows = conn.execute(
            "SELECT DISTINCT cudaId FROM TARGET_INFO_CUDA_DEVICE "
            "WHERE pid IN (SELECT pid FROM PROCESSES WHERE globalPid = ?) "
            "AND cudaId IS NOT NULL",
            (global_pid,),
        ).fetchall()
        devices.update(int(row[0]) for row in rows)

    if device_id is not None:
        if devices and device_id not in devices:
            raise NsightTimelineError(
                "requested device_id %d is absent from target process"
                % device_id
            )
        return int(device_id)
    if len(devices) != 1:
        raise NsightTimelineError(
            "expected one CUDA device for target process, found %s; "
            "pass --device-id for a multi-GPU trace" % sorted(devices)
        )
    return devices.pop()


def _relative_seconds(timestamp_ns: int, origin_ns: int) -> float:
    return (int(timestamp_ns) - origin_ns) / 1_000_000_000.0


def _event_bounds(
    row: Mapping[str, Any], origin_ns: int
) -> Tuple[float, float]:
    try:
        start = row["start"]
        end = row["end"]
    except (KeyError, IndexError):
        raise NsightTimelineError("CUDA activity has no start/end columns")
    if start is None or end is None:
        raise NsightTimelineError(
            "CUDA activity has a missing start/end timestamp"
        )
    start_sec = _relative_seconds(start, origin_ns)
    end_sec = _relative_seconds(end, origin_ns)
    if end_sec < start_sec:
        raise NsightTimelineError("CUDA activity has end before start")
    return start_sec, end_sec


def _merge_duration(intervals: Iterable[Tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left


def _bin_edges(
    event_ranges: Sequence[Tuple[float, float]], width: float
) -> range:
    if not event_ranges:
        return range(0)
    minimum = min(start for start, _ in event_ranges)
    maximum = max(end for _, end in event_ranges)
    first = math.floor(minimum / width)
    # Include the bin beginning at an exact completion boundary.  This makes
    # bins half-open and assigns an event ending at 0.1 to [0.1, 0.2).
    last = math.floor(maximum / width)
    return range(first, last + 1)


def extract_timeline(
    sqlite_path: Path,
    telemetry_path: Path,
    output_path: Path,
    bin_ms: int = 100,
    device_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Return and write a telemetry receipt enriched with CUDA activity."""
    if bin_ms <= 0:
        raise NsightTimelineError("bin_ms must be positive")
    receipt = json.loads(telemetry_path.read_text())
    target_pid = receipt.get("target_pid")
    if not isinstance(target_pid, int) or isinstance(target_pid, bool):
        raise NsightTimelineError(
            "telemetry receipt must contain integer top-level target_pid"
        )

    width = bin_ms / 1000.0
    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        origin_ns = _find_origin_ns(conn)
        global_pid = _resolve_global_pid(conn, target_pid)
        selected_device = _target_devices(conn, global_pid, device_id)

        kernels: List[Dict[str, float]] = []
        kernel_ranges: List[Tuple[float, float]] = []
        if _has_table(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
            rows = conn.execute(
                "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE globalPid = ? AND deviceId = ? ORDER BY start, end",
                (global_pid, selected_device),
            ).fetchall()
            for row in rows:
                start_sec, end_sec = _event_bounds(row, origin_ns)
                kernels.append({"start_sec": start_sec, "end_sec": end_sec})
                kernel_ranges.append((start_sec, end_sec))

        transfers: List[Dict[str, Any]] = []
        transfer_ranges: List[Tuple[float, float]] = []
        if _has_table(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            rows = conn.execute(
                "SELECT start, end, bytes, copyKind, srcKind, dstKind "
                "FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE globalPid = ? AND deviceId = ? ORDER BY start, end",
                (global_pid, selected_device),
            ).fetchall()
            for row in rows:
                copy_kind = int(row["copyKind"])
                direction = COPY_DIRECTIONS.get(copy_kind)
                if direction is None:
                    raise NsightTimelineError(
                        "unsupported CUDA memcpy copyKind %d for target "
                        "process; refusing to discard it" % copy_kind
                    )
                start_sec, end_sec = _event_bounds(row, origin_ns)
                transfers.append(
                    {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "bytes": int(row["bytes"]),
                        "direction": direction,
                        "source_memory_kind": _memory_kind(
                            conn, row["srcKind"]
                        ),
                        "destination_memory_kind": _memory_kind(
                            conn, row["dstKind"]
                        ),
                    }
                )
                transfer_ranges.append((start_sec, end_sec))

        bins: List[Dict[str, Any]] = []
        for index in _bin_edges(kernel_ranges + transfer_ranges, width):
            start = index * width
            end = start + width
            active = _merge_duration(
                (max(start, left), min(end, right))
                for left, right in kernel_ranges
                if right > start and left < end
            )
            bytes_by_direction = {"H2D": 0, "D2H": 0, "D2D": 0}
            for transfer in transfers:
                completion = transfer["end_sec"]
                if start <= completion < end:
                    bytes_by_direction[transfer["direction"]] += transfer[
                        "bytes"
                    ]
            bins.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "kernel_active_percent": round(100.0 * active / width, 6),
                    "h2d_bytes": bytes_by_direction["H2D"],
                    "d2h_bytes": bytes_by_direction["D2H"],
                    "d2d_bytes": bytes_by_direction["D2D"],
                }
            )

        totals = {
            "kernel_active_sec": _merge_duration(kernel_ranges),
            "kernel_count": len(kernels),
            "transfer_count": len(transfers),
            "h2d_bytes": sum(
                t["bytes"] for t in transfers if t["direction"] == "H2D"
            ),
            "d2h_bytes": sum(
                t["bytes"] for t in transfers if t["direction"] == "D2H"
            ),
            "d2d_bytes": sum(
                t["bytes"] for t in transfers if t["direction"] == "D2D"
            ),
        }

    receipt["cuda_trace"] = {
        "source": "Nsight Systems / CUPTI",
        "target_pid": target_pid,
        "global_pid": global_pid,
        "device_id": selected_device,
        "origin_ns": origin_ns,
        "bin_width_ms": bin_ms,
        "bins": bins,
        "transfers": transfers,
        "kernels": kernels,
        "totals": totals,
        "transfer_bin_assignment": "completion_time",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-ms", type=int, default=100)
    parser.add_argument("--device-id", type=int)
    args = parser.parse_args(argv)
    try:
        extract_timeline(
            args.sqlite,
            args.telemetry,
            args.output,
            bin_ms=args.bin_ms,
            device_id=args.device_id,
        )
    except (
        OSError,
        sqlite3.Error,
        json.JSONDecodeError,
        NsightTimelineError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
