"""A clean candidate checkout must never silently become the CPU reference."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_reference import validate_reference


def test_reference_requires_exact_clean_commit(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Benchmark test")
    git("config", "user.email", "benchmark@example.invalid")
    source = tmp_path / "calculation.py"
    source.write_text("result = 1\n")
    git("add", "calculation.py")
    git("commit", "-qm", "original")
    original = git("rev-parse", "HEAD")
    assert validate_reference(tmp_path, original) == original
    with pytest.raises(ValueError, match="40-character"):
        validate_reference(tmp_path, "HEAD")
    source.write_text("result = 2\n")
    with pytest.raises(ValueError, match="clean"):
        validate_reference(tmp_path, original)
    git("commit", "-qam", "changed calculation")
    with pytest.raises(ValueError, match="pinned"):
        validate_reference(tmp_path, original)
    changed = git("rev-parse", "HEAD")
    (tmp_path / "replacement.py").write_text("result = 3\n")
    with pytest.raises(ValueError, match="clean"):
        validate_reference(tmp_path, changed)
