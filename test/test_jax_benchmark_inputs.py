"""Tests for the portable JAX benchmark input preparation CLI."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from tools import prepare_jax_benchmark_inputs as prepare


def _fixture_bank(path: Path, rows: int = 40) -> None:
    with h5py.File(path, "w") as bank:
        bank["mass1"] = np.linspace(1.4, 8.0, rows)
        bank["mass2"] = np.full(rows, 1.3)
        bank["spin1z"] = np.zeros(rows)
        bank["spin2z"] = np.zeros(rows)
        bank["f_lower"] = np.full(rows, 30.0)


def _args(tmp_path: Path, frame: Path, bank_xml: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path / "outputs",
        python="python",
        reference_source=tmp_path / "reference",
        reference_revision=prepare.DEFAULT_REFERENCE_REVISION,
        frame_file=frame,
        frame_sha256=hashlib.sha256(frame.read_bytes()).hexdigest(),
        bank_xml=bank_xml,
        bank_xml_sha256=hashlib.sha256(bank_xml.read_bytes()).hexdigest(),
        sizes=(2, 4),
        plan=False,
    )


def test_select_rows_is_nested_and_physically_unique(tmp_path):
    source = tmp_path / "bank.hdf"
    _fixture_bank(source)
    rows = prepare.select_rows(source, (2, 4))
    assert rows["2"] == sorted(rows["2"])
    assert rows["4"] == sorted(rows["4"])
    assert set(rows["2"]).issubset(rows["4"])


def test_compressed_output_rejects_skipped_waveforms(tmp_path):
    compressed = tmp_path / "compressed.hdf"
    with h5py.File(compressed, "w") as output:
        output["template_hash"] = np.arange(2)
        output.create_group("compressed_waveforms").create_group("0")
    with pytest.raises(ValueError, match="stores 1 waveforms"):
        prepare._validate_compressed_output(compressed, expected_rows=2)


def test_prepare_mocks_reference_conversion_and_compression(tmp_path):
    frame = tmp_path / "frame.gwf"
    frame.write_bytes(b"frame")
    bank_xml = tmp_path / "bank.xml.gz"
    bank_xml.write_bytes(b"xml")
    args = _args(tmp_path, frame, bank_xml)
    calls = []

    run_kwargs = []

    def fake_run(command, check=True, **kwargs):
        calls.append(command)
        run_kwargs.append(kwargs)
        if len(command) > 2 and command[1] == "-c":
            return subprocess.CompletedProcess(
                command, 0,
                stdout=json.dumps({
                    "python": {"executable": "/fake/python", "version": "3.12.0",
                                "implementation": "CPython"},
                    "platform": "fake-platform",
                    "dependencies": {"pycbc": "1.2.3", "numpy": "9.8.7"},
                }),
                stderr="",
            )
        if command[1].endswith("pycbc_coinc_bank2hdf"):
            _fixture_bank(Path(command[command.index("--output-file") + 1]))
        else:
            bank = Path(command[command.index("--bank-file") + 1])
            output = Path(command[command.index("--output") + 1])
            with h5py.File(bank, "r") as source, h5py.File(output, "w") as compressed:
                rows = len(source["mass1"])
                compressed["template_hash"] = np.arange(rows)
                waveforms = compressed.create_group("compressed_waveforms")
                for index in range(rows):
                    waveforms.create_group(str(index))

    reference = {"source": str(args.reference_source),
                 "revision": args.reference_revision, "clean": True}
    with patch.object(prepare, "validate_reference", return_value=reference), \
            patch.object(prepare.subprocess, "run", side_effect=fake_run):
        manifest = prepare.prepare(args)

    assert len(calls) == 4
    assert manifest["runtime"]["dependencies"]["pycbc"] == "1.2.3"
    assert manifest["selection_runtime"]["dependencies"]["numpy"] == np.__version__
    tool_calls = calls[1:]
    assert "--approximant" not in tool_calls[0]
    assert "--compression-algorithm" in tool_calls[1]
    assert tool_calls[1][tool_calls[1].index("--psd-model") + 1] == (
        "aLIGOZeroDetHighPower"
    )
    assert tool_calls[1][tool_calls[1].index("--nprocesses") + 1] == "1"
    assert tool_calls[1][tool_calls[1].index("--precision") + 1] == "single"
    tool_kwargs = run_kwargs[1:]
    assert all(kwargs["cwd"] == str(args.reference_source.resolve()) for kwargs in tool_kwargs)
    assert all(kwargs["env"]["PYTHONPATH"] == str(args.reference_source.resolve())
               for kwargs in tool_kwargs)
    assert all(kwargs["env"]["PYTHONHASHSEED"] == "0" for kwargs in tool_kwargs)
    assert manifest["selection"]["seed"] == 20260919
    assert manifest["options"]["psd_model"] == "aLIGOZeroDetHighPower"
    assert manifest["options"]["nprocesses"] == 1
    assert manifest["banks"]["4"]["source_rows"]
    assert Path(manifest["banks"]["4"]["compressed_path"]).is_file()
    saved = json.loads((args.output_dir / "manifest.json").read_text())
    assert saved["inputs"]["frame_sha256"] == args.frame_sha256


def test_existing_output_fails_before_subprocess(tmp_path):
    frame = tmp_path / "frame.gwf"
    frame.write_bytes(b"frame")
    bank_xml = tmp_path / "bank.xml.gz"
    bank_xml.write_bytes(b"xml")
    args = _args(tmp_path, frame, bank_xml)
    args.output_dir.mkdir()
    (args.output_dir / "o2-subset-2.hdf").write_bytes(b"existing")
    reference = {"source": str(args.reference_source),
                 "revision": args.reference_revision, "clean": True}
    with patch.object(prepare, "validate_reference", return_value=reference), \
            patch.object(prepare.subprocess, "run") as launch:
        with pytest.raises(FileExistsError):
            prepare.prepare(args)
    launch.assert_not_called()


def test_output_inside_reference_checkout_is_rejected(tmp_path):
    frame = tmp_path / "frame.gwf"
    frame.write_bytes(b"frame")
    bank_xml = tmp_path / "bank.xml.gz"
    bank_xml.write_bytes(b"xml")
    args = _args(tmp_path, frame, bank_xml)
    args.output_dir = args.reference_source / "outputs"
    with patch.object(prepare, "validate_reference"):
        with pytest.raises(ValueError, match="outside the reference"):
            prepare.prepare(args)
    assert not args.output_dir.exists()


def test_plan_does_not_run_or_download(tmp_path):
    frame = tmp_path / "frame.gwf"
    frame.write_bytes(b"frame")
    existing_xml = tmp_path / "bank.xml.gz"
    existing_xml.write_bytes(b"xml")
    args = _args(tmp_path, frame, existing_xml)
    args.bank_xml = None
    args.plan = True
    args.bank_xml_sha256 = "0" * 64
    reference = {"source": str(args.reference_source),
                 "revision": args.reference_revision, "clean": True}
    with patch.object(prepare, "validate_reference", return_value=reference), \
            patch.object(prepare.subprocess, "run") as launch, patch.object(
            prepare, "urlretrieve") as download:
        result = prepare.prepare(args)
    launch.assert_not_called()
    download.assert_not_called()
    assert not args.output_dir.exists()
    assert result["plan"] is True
    assert len(result["commands"]["compression"]) == 2
