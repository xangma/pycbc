#!/usr/bin/env python3
"""Verify a reviewed split archive and safely restore it without executing it."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import tarfile
import tempfile

EXPECTED_ARCHIVE = 'a86204f5163c40f9ce011406cbc9f025209b48bbc9d95b18d0784bdce4772375'
EXPECTED_BYTES = 224392604
EXPECTED_ROOT = 'pycbc-torch-profile-20260907-r3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(name):
    p = PurePosixPath(name)
    require(p.parts and not p.is_absolute() and '..' not in p.parts
            and str(p) == name and '\\' not in name and '\x00' not in name,
            'Unsafe path: ' + name)
    return p


def check_publication(bundle):
    listed = {}
    for line in (bundle / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        safe_path(name)
        require(name not in listed, 'Duplicate publication filename')
        listed[name] = sha
    actual = set()
    for p in bundle.rglob('*'):
        require(not p.is_symlink(), 'Publication links are not allowed')
        require(p.is_file() or p.is_dir(), 'Publication special file')
        if p.is_file() and p.relative_to(bundle).as_posix() != 'SHA256SUMS':
            actual.add(p.relative_to(bundle).as_posix())
    require(set(listed) == actual, 'Publication file inventory differs')
    for name, sha in listed.items():
        require(digest(bundle / name) == sha, 'Publication hash differs: ' + name)
    return listed


def member_row(m):
    kind = ('file' if m.isfile() else 'directory' if m.isdir()
            else 'symlink' if m.issym() else 'unsupported')
    return dict(path=m.name, type=kind, size=m.size, mode=m.mode,
                uid=m.uid, gid=m.gid, uname=m.uname, gname=m.gname,
                mtime=m.mtime, linkname=m.linkname, pax_headers=m.pax_headers)


def validate_members(members, inventory):
    require(inventory['root'] == EXPECTED_ROOT, 'Unexpected archive root')
    expected = {r['path']: r for r in inventory['members']}
    require(len(expected) == inventory['member_count'], 'Duplicate inventory member')
    seen, links = set(), {}
    for m in members:
        path = safe_path(m.name)
        require(path.parts[0] == EXPECTED_ROOT, 'Wrong archive root')
        require(m.name not in seen, 'Duplicate archive member')
        seen.add(m.name)
        require(m.isfile() or m.isdir() or m.issym(), 'Hard links/special files rejected')
        row = expected.get(m.name)
        require(row is not None and all(row[k] == v for k, v in member_row(m).items()),
                'Archive metadata differs: ' + m.name)
        if m.issym():
            target = PurePosixPath(m.linkname)
            require(m.linkname and not target.is_absolute() and '\\' not in m.linkname
                    and '\x00' not in m.linkname, 'Unsafe symlink target')
            resolved = posixpath.normpath(str(path.parent / target))
            safe_path(resolved)
            require(resolved.startswith(EXPECTED_ROOT + '/') and resolved in expected
                    and expected[resolved]['type'] == 'file',
                    'Symlink must target an inventoried internal regular file')
            links[m.name] = m.linkname
    require(seen == set(expected), 'Archive member inventory differs')
    for name in seen:
        for parent in PurePosixPath(name).parents:
            require(str(parent) not in links, 'Archive member below a symlink')
    return expected, links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--restore', type=Path, help='Fresh destination in an existing real parent')
    mode.add_argument('--check-only', action='store_true', help='Check without retaining restored files')
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    listed = check_publication(bundle)
    parts = json.loads((bundle / 'archive-parts.json').read_text())
    inventory = json.loads((bundle / 'archive-inventory.json').read_text())
    require(parts['archive_sha256'] == inventory['archive_sha256'] == EXPECTED_ARCHIVE
            and parts['archive_bytes'] == inventory['archive_bytes'] == EXPECTED_BYTES,
            'Archive pin differs from reviewed evidence')
    require(len(parts['parts']) == 9 and [r['name'] for r in parts['parts']] ==
            [f'evidence.tar.gz.part{i:03d}' for i in range(9)], 'Part order differs')
    destination = args.restore.absolute() if args.restore else None
    if destination is not None:
        require(not destination.exists() and not destination.is_symlink(), 'Destination must be fresh')
        require(destination.parent.is_dir() and destination.parent.resolve() == destination.parent,
                'Restore parent must exist without symlink aliases')
    # Reassemble in a private temporary file before creating the destination.
    with tempfile.TemporaryFile() as archive:
        combined, total = hashlib.sha256(), 0
        for part in parts['parts']:
            safe_path(part['name'])
            p = bundle / part['name']
            require(listed.get(part['name']) == part['sha256'] and p.stat().st_size == part['bytes'],
                    'Part hash/length differs')
            with p.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    archive.write(block)
                    combined.update(block)
                    total += len(block)
        require(combined.hexdigest() == EXPECTED_ARCHIVE and total == EXPECTED_BYTES,
                'Reassembled archive differs')
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode='r:gz') as stream:
            members = stream.getmembers()
            expected, links = validate_members(members, inventory)
            if destination is not None:
                destination.mkdir()
                archive.seek(0)
                with (destination / 'evidence.tar.gz').open('xb') as out:
                    shutil.copyfileobj(archive, out)
                require(digest(destination / 'evidence.tar.gz') == EXPECTED_ARCHIVE,
                        'Restored archive differs')
            for m in members:
                if m.isdir() and destination is not None:
                    (destination / m.name).mkdir(parents=True, exist_ok=True)
                if not m.isfile():
                    continue
                h = hashlib.sha256()
                out = None
                if destination is not None:
                    path = destination / m.name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    out = path.open('xb')
                try:
                    with stream.extractfile(m) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            h.update(block)
                            if out is not None:
                                out.write(block)
                finally:
                    if out is not None:
                        out.close()
                require(h.hexdigest() == expected[m.name]['sha256'], 'Member hash differs: ' + m.name)
                if destination is not None:
                    path.chmod(m.mode & 0o777)
            if destination is not None:
                for name, target in links.items():
                    (destination / name).symlink_to(target)
                for m in reversed(members):
                    if m.isdir():
                        (destination / m.name).chmod(m.mode & 0o777)
                root = destination / EXPECTED_ROOT
                actual = {p.relative_to(destination).as_posix() for p in root.rglob('*')}
                actual.add(EXPECTED_ROOT)
                require(actual == set(expected), 'Restored tree inventory differs')
                for name, target in links.items():
                    p = destination / name
                    require(p.is_symlink() and str(p.readlink()) == target,
                            'Restored symlink differs')
                prior = destination / 'previous-cpu'
                prior.mkdir()
                for name in ('triggers.hdf', 'receipt.json', 'runtime.json'):
                    shutil.copyfile(bundle / 'previous-cpu' / name, prior / name)
                    require(digest(prior / name) == listed['previous-cpu/' + name],
                            'Supplemental previous CPU file differs')
    print(json.dumps(dict(state='pass', archive_sha256=EXPECTED_ARCHIVE, archive_bytes=total,
                          publication_files=len(listed) + 1, parts=9,
                          archive_member_count=len(expected), file_count=inventory['file_count'],
                          directory_count=inventory['directory_count'], symlink_count=len(links),
                          restored_root=str(destination) if destination is not None else None,
                          all_member_hashes_match=True, all_publication_hashes_match=True,
                          symlink_targets_internal=True,
                          source_executable_bits_preserved=True if destination is not None else None,
                          supplemental_previous_cpu_files=3, science_executed=False), indent=2))


if __name__ == '__main__':
    main()
