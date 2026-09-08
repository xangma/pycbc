from pathlib import Path
import hashlib
root = Path(__file__).resolve().parent
expected = {}
for line in (root / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split("  ", 1)
    path = Path(name)
    assert not path.is_absolute() and ".." not in path.parts, name
    assert name not in expected, name
    expected[name] = digest
actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
assert actual == set(expected), (actual - set(expected), set(expected) - actual)
for name, digest in expected.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
print(f"PASS: {len(expected)} files; exact inventory and SHA-256 hashes verified")
