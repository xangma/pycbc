#!/usr/bin/env python3
"""Verify this publication and safely restore its unchanged evidence archive.

Uses only the standard library. Never imports or executes archived code.
Preserves file bytes and paths, not tar ownership, timestamps or permissions.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


ARCHIVE_NAME = 'executable-remote-evidence.tar.gz'
EXPECTED_ARCHIVE = (
    'f502e162bcc61e5636f5764572f09b6c3ec1e0ed0de617d15eefffdd0a3c9754'
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def safe_path(name, directory=False):
    """Preserve the original '.' root and './...' archive spelling exactly."""
    if name == '.':
        require(directory, 'Archive root must be a directory')
        return PurePosixPath('.')
    require(name.startswith('./'), 'Archive path lacks ./ prefix: ' + name)
    path = PurePosixPath(name[2:])
    require(path.parts and not path.is_absolute()
            and '..' not in path.parts and '\\' not in name and ':' not in name
            and not any(ord(char) < 32 or ord(char) == 127 for char in name)
            and './' + str(path) == name,
            'Unsafe or noncanonical archive path: ' + name)
    return path


def publication_hashes(bundle):
    manifest = bundle / 'SHA256SUMS'
    require(manifest.is_file() and not manifest.is_symlink(),
            'SHA256SUMS must be a regular file')
    listed = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', expected) is not None,
                'Invalid SHA256SUMS digest')
        require(name not in {'', '.', '..', 'SHA256SUMS'}
                and '/' not in name and '\\' not in name and ':' not in name
                and not any(ord(char) < 32 or ord(char) == 127 for char in name)
                and name not in listed, 'Unsafe or duplicate publication name')
        listed.add(name)
        path = bundle / name
        require(path.is_file() and not path.is_symlink(),
                'Publication member must be a regular file: ' + name)
        require(digest(path) == expected, 'Publication hash mismatch: ' + name)
    actual = set()
    for path in bundle.iterdir():
        require(path.is_file() and not path.is_symlink(),
                'Unexpected publication directory, link or special file')
        if path.name != 'SHA256SUMS':
            actual.add(path.name)
    require(actual == listed, 'Publication inventory differs')
    return listed


def validate_members(members, inventory):
    expected = {row['path']: row for row in inventory['members']}
    require(len(expected) == len(inventory['members']) == inventory['member_count'],
            'Duplicate or incomplete archive inventory')
    seen = set()
    paths = {}
    for member in members:
        require(member.isfile() or member.isdir(),
                'Archive links and special files are not allowed')
        path = safe_path(member.name, member.isdir())
        require(member.name not in seen and path not in paths,
                'Duplicate archive member or path alias')
        seen.add(member.name)
        paths[path] = member
        row = expected.get(member.name)
        require(row is not None, 'Unlisted archive member: ' + member.name)
        actual = dict(path=member.name,
                      type='file' if member.isfile() else 'directory',
                      size=member.size, mode=member.mode,
                      uid=member.uid, gid=member.gid, uname=member.uname,
                      gname=member.gname, mtime=member.mtime,
                      linkname=member.linkname, pax_headers=member.pax_headers,
                      tar_type=member.type.decode('ascii'))
        require(actual == {key: value for key, value in row.items()
                           if key != 'sha256'},
                'Archive metadata differs: ' + member.name)
        if member.isfile():
            require(re.fullmatch('[0-9a-f]{64}', row.get('sha256', ''))
                    is not None, 'Invalid archived file hash')
    require(seen == set(expected), 'Archive member set differs')
    require(PurePosixPath('.') in paths
            and paths[PurePosixPath('.')].isdir(), 'Missing archive root')
    for path in paths:
        for parent in path.parents:
            require(parent in paths and paths[parent].isdir(),
                    'Missing directory or file/directory conflict')
    files = [member for member in members if member.isfile()]
    require(len(files) == inventory['file_count'] == 190
            and len(members) - len(files) == inventory['directory_count'] == 19
            and sum(member.size for member in files)
            == inventory['uncompressed_file_bytes'] == 33368555,
            'Archive totals differ')
    return expected


def restore(bundle, destination):
    publication_hashes(bundle)
    inventory = json.loads((bundle / 'archive-inventory.json').read_text())
    archive = bundle / ARCHIVE_NAME
    require(inventory['archive'] == ARCHIVE_NAME and inventory['root'] == '.'
            and archive.stat().st_size == inventory['archive_size'] == 12223085
            and digest(archive) == EXPECTED_ARCHIVE == inventory['archive_sha256'],
            'Archive differs from the reviewed original')
    audit = json.loads((bundle / 'executable-terminal-audit.json').read_text())
    inventory_hashes = {row['path'][2:]: row['sha256']
                        for row in inventory['members'] if row['type'] == 'file'}
    require(inventory_hashes == audit['evidence_sha256'],
            'Archive differs from the independent terminal audit')
    lines = ''.join(f'{sha}  ./{name}\n'
                    for name, sha in sorted(inventory_hashes.items()))
    require(lines == (bundle / 'archive-members.sha256').read_text(),
            'Archive hash inventory differs')
    destination = destination.absolute()
    require(not destination.exists() and not destination.is_symlink(),
            'Restore destination must be fresh')
    require(destination.parent.is_dir()
            and destination.parent.resolve() == destination.parent,
            'Restore parent must exist and contain no symlink aliases')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        expected = validate_members(members, inventory)
        # No destination is created until all paths and metadata are validated.
        destination.mkdir(mode=0o700)
        for member in members:
            target = destination / safe_path(member.name, member.isdir())
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
            else:
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out)
                require(digest(target) == expected[member.name]['sha256'],
                        'Restored hash mismatch: ' + member.name)
    restored = {'.'} | {'./' + path.relative_to(destination).as_posix()
                        for path in destination.rglob('*')}
    require(restored == set(expected), 'Restored member set differs')
    return dict(state='pass', archive_sha256=EXPECTED_ARCHIVE,
                member_count=len(expected), file_count=inventory['file_count'],
                directory_count=inventory['directory_count'],
                restored_root=str(destination), safe_paths=True,
                links_or_special_files=0, original_archive_paths_preserved=True,
                all_restored_file_hashes_match=True,
                independent_terminal_audit_hashes_match=True,
                publication_file_hashes_match=True, science_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', type=Path, required=True)
    args = parser.parse_args()
    result = restore(Path(__file__).resolve().parent, args.restore)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
