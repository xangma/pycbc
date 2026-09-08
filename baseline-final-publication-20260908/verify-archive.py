"""Verify the complete sealed file inventory without importing acquisition code."""
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    expected = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts, name
        assert name not in expected and len(sha) == 64, name
        expected[name] = sha
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root / 'SHA256SUMS'}
    assert actual == set(expected), {'missing': sorted(set(expected) - actual),
                                     'extra': sorted(actual - set(expected))}
    for name, sha in expected.items():
        path = root / name
        assert not path.is_symlink(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, name
    print(f'PASS: {len(expected)} files match the sealed SHA256 inventory')


if __name__ == '__main__':
    main()
