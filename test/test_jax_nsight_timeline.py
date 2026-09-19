"""Focused tests for process-scoped Nsight Systems CUDA extraction."""

import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extract_jax_nsight_timeline",
    ROOT / "tools" / "extract_jax_nsight_timeline.py",
)
nsight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nsight)


TARGET_PID = 123
TARGET_GLOBAL_PID = TARGET_PID << 24
OTHER_GLOBAL_PID = 999 << 24
ORIGIN_NS = 1_000_000_000


def make_trace(path, *, marker=True, two_devices=False):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER, name TEXT);
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER NOT NULL, end INTEGER, eventType INTEGER NOT NULL,
            text TEXT, textId INTEGER, globalTid INTEGER
        );
        CREATE TABLE ENUM_CUDA_MEM_KIND (
            id INTEGER PRIMARY KEY, name TEXT, label TEXT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER,
            bytes INTEGER, copyKind INTEGER, srcKind INTEGER, dstKind INTEGER
        );
        """
    )
    conn.executemany(
        "INSERT INTO PROCESSES VALUES (?, ?, ?)",
        [(TARGET_GLOBAL_PID, TARGET_PID, "python"),
         (OTHER_GLOBAL_PID, 999, "other")],
    )
    conn.executemany(
        "INSERT INTO ENUM_CUDA_MEM_KIND VALUES (?, ?, ?)",
        [(0, "PAGEABLE", "Pageable"), (1, "PINNED", "Pinned"),
         (2, "DEVICE", "Device")],
    )
    conn.execute("INSERT INTO StringIds VALUES (1, ?)", (nsight.ORIGIN_MARK,))
    if marker:
        # Exercise the StringIds join, rather than the denormalized text field.
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, NULL, 1, NULL, 1, ?)",
            (ORIGIN_NS, TARGET_PID),
        )
    # Two overlapping kernels: their union is 80 ms in [0, .1) and 20 ms
    # in [.1, .2), rather than counting 100 ms in the first bin.
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
        [(1_020_000_000, 1_080_000_000, 0, TARGET_GLOBAL_PID),
         (1_050_000_000, 1_120_000_000, 0, TARGET_GLOBAL_PID),
         (1_000_000_000, 1_200_000_000, 0, OTHER_GLOBAL_PID)],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)",
        [(1_010_000_000, 1_050_000_000, 0, TARGET_GLOBAL_PID,
          10, 1, 1, 2),
         # Completion exactly on a bin edge belongs to the next bin.
         (1_080_000_000, 1_100_000_000, 0, TARGET_GLOBAL_PID,
          20, 2, 2, 0),
         (1_120_000_000, 1_180_000_000, 0, TARGET_GLOBAL_PID,
          30, 8, 2, 2),
         (1_010_000_000, 1_020_000_000, 0, OTHER_GLOBAL_PID,
          999, 1, 1, 2)],
    )
    if two_devices:
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
            (1_300_000_000, 1_310_000_000, 1, TARGET_GLOBAL_PID),
        )
    conn.commit()
    conn.close()


def write_receipt(path, **extra):
    data = {"schema_version": 2, "target_pid": TARGET_PID, "telemetry": []}
    data.update(extra)
    path.write_text(json.dumps(data))


def test_process_filter_alignment_overlap_and_completion_bins(tmp_path):
    db = tmp_path / "trace.sqlite"
    receipt = tmp_path / "telemetry.json"
    output = tmp_path / "enriched.json"
    make_trace(db)
    write_receipt(receipt, untouched={"yes": True})

    result = nsight.extract_timeline(db, receipt, output, bin_ms=100)
    trace = result["cuda_trace"]
    assert result["untouched"] == {"yes": True}
    assert trace["target_pid"] == TARGET_PID
    assert trace["global_pid"] == TARGET_GLOBAL_PID
    assert trace["origin_ns"] == ORIGIN_NS
    assert trace["device_id"] == 0
    assert trace["kernels"] == [
        {"start_sec": 0.02, "end_sec": 0.08},
        {"start_sec": 0.05, "end_sec": 0.12},
    ]
    assert [b["kernel_active_percent"] for b in trace["bins"]] == [80.0, 20.0]
    assert trace["bins"][0]["h2d_bytes"] == 10
    assert trace["bins"][1]["d2h_bytes"] == 20
    assert trace["bins"][1]["d2d_bytes"] == 30
    assert trace["totals"]["h2d_bytes"] == 10
    assert trace["totals"]["d2h_bytes"] == 20
    assert trace["totals"]["d2d_bytes"] == 30
    assert trace["totals"]["kernel_active_sec"] == pytest.approx(0.1)
    assert json.loads(output.read_text())["cuda_trace"] == trace


def test_missing_marker_is_rejected(tmp_path):
    db = tmp_path / "trace.sqlite"
    receipt = tmp_path / "telemetry.json"
    make_trace(db, marker=False)
    write_receipt(receipt)
    with pytest.raises(nsight.NsightTimelineError, match="exactly one"):
        nsight.extract_timeline(db, receipt, tmp_path / "out.json")


def test_ambiguous_devices_require_explicit_selection(tmp_path):
    db = tmp_path / "trace.sqlite"
    receipt = tmp_path / "telemetry.json"
    make_trace(db, two_devices=True)
    write_receipt(receipt)
    with pytest.raises(nsight.NsightTimelineError, match="one CUDA device"):
        nsight.extract_timeline(db, receipt, tmp_path / "out.json")
    result = nsight.extract_timeline(
        db, receipt, tmp_path / "out.json", device_id=1
    )
    assert result["cuda_trace"]["device_id"] == 1


def test_unknown_copy_kind_is_rejected_without_discarding(tmp_path):
    db = tmp_path / "trace.sqlite"
    receipt = tmp_path / "telemetry.json"
    make_trace(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)",
        (1_100_000_000, 1_110_000_000, 0, TARGET_GLOBAL_PID, 7, 4, 3, 0),
    )
    conn.commit()
    conn.close()
    write_receipt(receipt)
    with pytest.raises(nsight.NsightTimelineError, match="copyKind 4"):
        nsight.extract_timeline(db, receipt, tmp_path / "out.json")
