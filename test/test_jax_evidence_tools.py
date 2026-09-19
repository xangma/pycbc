"""Evidence gates must fail closed independently of performance measurements."""

import base64
import json
import subprocess
import sys

import numpy as np
import pytest

from tools.benchmarking.evidence import array_error, git_snapshot, timed_command, write_receipt


def test_complex_errors_reject_phase_nonfinite_and_shape():
    reference = np.array([1 + 1j, 2 - 1j])
    assert not array_error(reference, reference * 1j, rtol=1e-3)["passed"]
    assert not array_error(reference, [np.nan, 1], rtol=1e-3)["passed"]
    assert not array_error(reference, [1], rtol=1e-3)["passed"]
    assert array_error([0], [0], rtol=0)["passed"]


def test_receipt_serialization_preserves_previous_evidence(tmp_path):
    path = tmp_path / "receipt.json"
    write_receipt(path, {"status": "passed"})
    with pytest.raises(ValueError):
        write_receipt(path, {"metric": float("nan")})
    assert json.loads(path.read_text()) == {"status": "passed"}


def test_git_snapshot_reconstructs_tracked_and_untracked_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    def git(*args, cwd=source, **kwargs):
        return subprocess.run(["git", *args], cwd=cwd, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)

    git("init")
    (source / "module.py").write_text("value = 1\n")
    git("add", "module.py")
    git("-c", "user.name=Evidence Test", "-c", "user.email=test@example.invalid",
        "-c", "core.hooksPath=/dev/null", "commit", "-m", "fixture")
    (source / "module.py").write_text("value = 2\n")
    (source / "provider.py").write_text('provider = "jax"\n')
    snapshot = git_snapshot(source)
    restored = tmp_path / "restored"
    git("clone", "--no-hardlinks", str(source), str(restored))
    git("apply", "-", cwd=restored, input=base64.b64decode(snapshot["tracked_patch_base64"]))
    for name, value in snapshot["untracked_source_base64"].items():
        (restored / name).write_bytes(base64.b64decode(value))
    for name in ("module.py", "provider.py"):
        assert (restored / name).read_bytes() == (source / name).read_bytes()
    assert snapshot["git_dirty"] is True


def test_process_timer_includes_launch_exit_and_output(tmp_path):
    output = tmp_path / "worker-output.txt"
    log = tmp_path / "worker.log"
    command = [sys.executable, "-c",
               'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("complete"); print("done")',
               str(output)]
    receipt = timed_command(command, log, cwd=tmp_path)
    assert receipt["returncode"] == 0
    assert output.read_text() == "complete"
    assert log.read_text().strip() == "done"
