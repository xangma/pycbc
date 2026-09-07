"""Restore this fixed evidence package; standard library, no archived execution."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

ARCHIVES = {
    'evidence.tar.gz': 'f62fc1c56c2c6820a444a0c73ec331b6b2fa54c7bafe3aefebca9f2bcbb320f1',
    'local-evidence.tar.gz': 'c733ee73da66f242a00b142122fc7cbc4ed609282f2ad22773be2aa7de04bdf3',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and '..' not in path.parts
            and str(path) == name and '\\' not in name and ':' not in name
            and not any(ord(c) < 32 or ord(c) == 127 for c in name),
            'Unsafe or noncanonical member path: ' + name)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sums = root / 'SHA256SUMS'
    require(sums.is_file() and not sums.is_symlink(), 'Invalid SHA256SUMS')
    listed = {}
    for line in sums.read_text().splitlines():
        expected, name = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', expected) and
                len(safe_path(name).parts) == 1 and name != 'SHA256SUMS'
                and name not in listed, 'Invalid publication inventory')
        listed[name] = expected
    actual = {}
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), 'Nonregular publication file')
        if path.name != 'SHA256SUMS':
            actual[path.name] = digest(path)
    require(actual == listed, 'Publication byte inventory differs')
    inventory = json.loads((root / 'archive-inventory.json').read_text())
    require(set(inventory) == set(ARCHIVES), 'Unexpected archive inventory')
    for name, expected in ARCHIVES.items():
        record = inventory[name]
        require(digest(root / name) == expected == record['sha256'] and
                (root / name).stat().st_size == record['bytes'], 'Archive differs')
        require(record['restore_subdirectory'] ==
                ('remote' if name == 'evidence.tar.gz' else 'local'), 'Invalid destination')
        with tarfile.open(root / name, 'r:gz') as archive:
            members = archive.getmembers()
            require(len(members) == len(record['files']) == record['file_count'],
                    'Archive member count differs')
            seen = set()
            for member in members:
                path = safe_path(member.name)
                require(member.isfile() and member.name not in seen,
                        'Link, special file or duplicate member')
                require(not any(str(p) in record['files'] for p in path.parents),
                        'File/directory conflict')
                value = archive.extractfile(member).read()
                require(record['files'].get(member.name) ==
                        dict(sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)),
                        'Archived bytes differ: ' + member.name)
                seen.add(member.name)
            require(seen == set(record['files']), 'Archive file inventory differs')
    destination = args.restore.absolute()
    require(not destination.exists() and not destination.is_symlink(),
            'Restore destination must be fresh')
    require(destination.parent.is_dir() and destination.parent.resolve() == destination.parent,
            'Restore parent must exist without symlink aliases')
    destination.mkdir(mode=0o700)
    for name, record in inventory.items():
        folder = destination / record['restore_subdirectory']
        folder.mkdir(mode=0o700)
        with tarfile.open(root / name, 'r:gz') as archive:
            for member in archive.getmembers():
                target = folder.joinpath(*safe_path(member.name).parts)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with target.open('xb') as output:
                    output.write(archive.extractfile(member).read())
                require(digest(target) == record['files'][member.name]['sha256'],
                        'Restored bytes differ')
    print(json.dumps(dict(state='pass', publication_files=len(listed) + 1,
                          restored_files=sum(r['file_count'] for r in inventory.values()),
                          archive_sha256=ARCHIVES, destination=str(destination),
                          science_executed=False), indent=2))


if __name__ == '__main__':
    main()
