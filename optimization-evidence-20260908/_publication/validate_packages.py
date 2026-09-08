"""Offline integrity, deterministic transport and filesystem reconstruction checks."""
import copy
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from inspect_inputs import HERE, digest
from build_packages import compressed, make_tar
from verify_package import verify


def reseal(root):
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (root/'SHA256SUMS').write_text(''.join(f'{digest(p.read_bytes())}  {p.relative_to(root).as_posix()}\n' for p in paths))


def negative_checks():
    source = HERE/'packages/torch-cpu-workspace-policy-20260908'
    results = []
    for case in ['changed_payload', 'missing_file', 'extra_file', 'symlink',
                 'traversal', 'file_directory_conflict', 'duplicate_manifest_key',
                 'existing_destination', 'archive_link', 'archive_traversal']:
        with tempfile.TemporaryDirectory(prefix='verifier-negative-', dir=HERE) as temp:
            root = Path(temp)/'package'
            shutil.copytree(source, root)
            manifest = json.loads((root/'manifest.json').read_text())
            destination = None
            if case == 'changed_payload':
                with (root/'supplement.tar.gz').open('ab') as stream:
                    stream.write(b'corrupt')
            elif case == 'missing_file':
                (root/'README.md').unlink()
            elif case == 'extra_file':
                (root/'unexpected.txt').write_text('extra')
            elif case == 'symlink':
                (root/'unexpected-link').symlink_to('README.md')
            elif case == 'duplicate_manifest_key':
                data = (root/'manifest.json').read_text()
                (root/'manifest.json').write_text('{"schema":"invalid",'+data[1:])
                reseal(root)
            elif case == 'existing_destination':
                destination = Path(temp)/'existing'
                destination.mkdir()
            elif case in ('archive_link','archive_traversal'):
                row = tarfile.TarInfo('../outside' if case=='archive_traversal' else 'unsafe-link')
                if case == 'archive_link':
                    row.type, row.linkname = tarfile.SYMTYPE, '../outside'
                raw = io.BytesIO()
                with tarfile.open(fileobj=raw, mode='w') as archive:
                    archive.addfile(row, io.BytesIO())
                data = raw.getvalue()
                (root/'supplement.tar.gz').write_bytes(compressed(data))
                manifest['supplement']['original'] = dict(bytes=len(data),sha256=digest(data))
                inventory = json.loads((root/'archive-inventory.json').read_text())
                inventory['supplement']['members'] = [{}]
                (root/'archive-inventory.json').write_text(json.dumps(inventory))
                (root/'manifest.json').write_text(json.dumps(manifest))
                reseal(root)
            else:
                first = next(iter(manifest['files'].values()))
                if case == 'traversal':
                    manifest['files']['../outside'] = first
                else:
                    manifest['files']['conflict'] = first
                    manifest['files']['conflict/child'] = first
                (root/'manifest.json').write_text(json.dumps(manifest))
                reseal(root)
            try:
                verify(root, destination)
            except ValueError as error:
                results.append(dict(case=case, state='pass', rejected=str(error)))
            else:
                raise AssertionError('Failed to reject: '+case)
            assert not (Path(temp)/'outside').exists()
    return results


def main():
    inputs = json.loads((HERE/'input-inspection.json').read_text())
    results = []
    for name, original in inputs['roots'].items():
        package = HERE/'packages'/name
        manifest = json.loads((package/'manifest.json').read_text())
        transport_count = 0
        for rel, row in manifest['transports'].items():
            data = (Path(original['source_root'])/rel).read_bytes()
            assert digest(compressed(data)) == digest((package/row['path']).read_bytes())
            transport_count += 1
        blobs = {}
        for rel, row in manifest['files'].items():
            if row['storage']['kind'] == 'supplement':
                blobs[row['storage']['member']] = (Path(original['source_root'])/rel).read_bytes()
        raw, _ = make_tar(blobs)
        assert digest(compressed(raw)) == digest((package/'supplement.tar.gz').read_bytes())
        with tempfile.TemporaryDirectory(prefix='reconstruction-', dir=HERE) as temp:
            restored = Path(temp)/name
            result = verify(package, restored)
            assert set(p.relative_to(restored).as_posix() for p in restored.rglob('*') if p.is_file()) == set(original['files'])
            for rel, row in original['files'].items():
                target, incoming = restored/rel, Path(original['source_root'])/rel
                assert digest(target.read_bytes()) == digest(incoming.read_bytes()) == row['sha256'], rel
                assert target.stat().st_size == row['bytes']
                assert target.stat().st_mode & 0o777 == row['mode']
            result['source_files_rehashed_after_reconstruction'] = len(original['files'])
            result['deterministic_transports_regenerated'] = transport_count + 1
            result['restored_files_equal_authoritative_selection'] = True
            result['file_modes_verified'] = True
            result['temporary_reconstruction_removed'] = True
            results.append(result)
        print(json.dumps(result), flush=True)
    negatives = negative_checks()
    report = dict(state='pass', packages=results, verifier_negative_checks=negatives,
        total_files=sum(r['reconstructed_files'] for r in results),
        total_bytes=sum(r['reconstructed_bytes'] for r in results),
        total_original_archives=sum(r['original_archives'] for r in results),
        total_original_seal_entries=sum(r['original_sealed_entries_verified'] for r in results),
        scope='All selected bytes reconstructed on disk, independently compared to the inputs and their initial hashes, then temporary reconstructed trees removed. No experiment code executed.',
        remote_access=False, science_executed=False)
    (HERE/'reconstruction-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('packages','verifier_negative_checks')},indent=2))


if __name__ == '__main__':
    main()
