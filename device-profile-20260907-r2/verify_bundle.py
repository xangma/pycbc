#!/usr/bin/env python3
"""Verify this reviewed publication and restore regular files into a fresh root.

Standard library only. Does not execute archived code or import PyCBC/Torch.
Archive metadata is inventoried; restoration checks file bytes, not ownership.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


EXPECTED_ARCHIVE = (
    '10298718066da7d1266a00cd874e87c7aaa0e7a8221cf1bfdd829196aa108870'
)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    listed = set()
    for line in (bundle / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        require(Path(name).name == name and name not in listed,
                'Unsafe or duplicate publication filename')
        listed.add(name)
        require(digest(bundle / name) == expected, 'Publication hash: ' + name)
    require(listed == {p.name for p in bundle.iterdir()
                       if p.is_file() and p.name != 'SHA256SUMS'},
            'Publication inventory differs')
    inventory = json.loads((bundle / 'archive-inventory.json').read_text())
    archive = bundle / 'remote-evidence.tar.gz'
    require(digest(archive) == EXPECTED_ARCHIVE == inventory['archive_sha256'],
            'Archive hash differs from reviewed evidence')
    expected = {r['path']: r for r in inventory['members']}
    require(len(expected) == inventory['member_count'], 'Duplicate inventory')
    destination = args.restore.absolute()
    require(not destination.exists() and not destination.is_symlink(),
            'Restore destination must be fresh')
    require(destination.parent.resolve() == destination.parent,
            'Restore parent must exist and contain no symlink aliases')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        seen = set()
        # Validate all paths and types before creating any output.
        for member in members:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and '..' not in path.parts
                    and path.parts and path.parts[0] == inventory['root']
                    and str(path) == member.name and '\\' not in member.name,
                    'Unsafe archive path: ' + member.name)
            require(member.name not in seen, 'Duplicate archive path')
            seen.add(member.name)
            require(member.isfile() or member.isdir(),
                    'Links and special files are not allowed')
            row = expected.get(member.name)
            require(row is not None and row['size'] == member.size
                    and row['type'] == ('file' if member.isfile() else 'directory'),
                    'Archive metadata mismatch: ' + member.name)
        require(seen == set(expected), 'Archive member set differs')
        destination.mkdir()
        for member in members:
            path = destination / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, path.open('xb') as out:
                    shutil.copyfileobj(source, out)
                require(digest(path) == expected[member.name]['sha256'],
                        'Restored hash mismatch: ' + member.name)
    restored = {str(p.relative_to(destination))
                for p in destination.rglob('*')}
    require(restored == set(expected), 'Restored member set differs')
    print(json.dumps(dict(
        state='pass', archive_sha256=EXPECTED_ARCHIVE,
        member_count=len(expected), file_count=inventory['file_count'],
        directory_count=inventory['directory_count'],
        restored_root=str(destination), safe_paths=True,
        links_or_special_files=0, all_restored_file_hashes_match=True,
        publication_file_hashes_match=True, science_executed=False), indent=2))


if __name__ == '__main__':
    main()
