"""Pin executable benchmark references without changing their calculations."""

import re
import subprocess
from pathlib import Path


def validate_reference(root: Path, expected_revision: str) -> str:
    """Require the declared immutable revision and a completely clean checkout."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_revision or ""):
        raise ValueError("reference revision must be an explicit full 40-character commit SHA")
    root = Path(root).resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True,
            stderr=subprocess.PIPE).strip()

    try:
        revision = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=all")
    except subprocess.CalledProcessError as exc:
        raise ValueError("reference source must be a readable git checkout") from exc
    if revision != expected_revision.lower():
        raise ValueError("reference HEAD differs from the explicitly pinned revision")
    if dirty:
        raise ValueError("reference checkout must be clean, including untracked files")
    return revision
