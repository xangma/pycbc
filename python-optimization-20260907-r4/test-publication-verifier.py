#!/usr/bin/env python3
"""Local standard-library checks of the publication verifier, without science."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT if (ROOT / 'verify_bundle.py').is_file() else ROOT / 'publication-prep'
SPEC = importlib.util.spec_from_file_location(
    'publication_verifier', BUNDLE / 'verify_bundle.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class VerifierSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads((BUNDLE / 'archive-inventory.json').read_text())
        with tarfile.open(BUNDLE / VERIFY.ARCHIVE_NAME, 'r:gz') as archive:
            cls.members = archive.getmembers()

    def test_original_metadata(self):
        self.assertEqual(len(VERIFY.validate_members(
            self.members, self.inventory)), 209)

    def test_unsafe_paths(self):
        for path in ['/escape', '../escape', './../escape', './a/../escape',
                     './a//b', './a/./b', './a/', './C:/escape',
                     './a\\b', './a\nb', './', '']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                VERIFY.safe_path(path)

    def test_root_file(self):
        with self.assertRaisesRegex(ValueError, 'root must be a directory'):
            VERIFY.safe_path('.')

    def test_links_and_special_files(self):
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE,
                     tarfile.BLKTYPE, tarfile.FIFOTYPE]:
            members = copy.deepcopy(self.members)
            members[1].type = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(
                    ValueError, 'links and special'):
                VERIFY.validate_members(members, self.inventory)

    def test_duplicate_member(self):
        members = self.members + [self.members[1]]
        with self.assertRaisesRegex(ValueError, 'Duplicate archive member'):
            VERIFY.validate_members(members, self.inventory)

    def test_duplicate_inventory(self):
        inventory = copy.deepcopy(self.inventory)
        inventory['members'].append(inventory['members'][1])
        with self.assertRaisesRegex(ValueError, 'Duplicate or incomplete'):
            VERIFY.validate_members(self.members, inventory)

    def test_metadata_change(self):
        members = copy.deepcopy(self.members)
        members[1].size += 1
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            VERIFY.validate_members(members, self.inventory)

    def test_file_directory_conflict(self):
        members = copy.deepcopy(self.members)
        inventory = copy.deepcopy(self.inventory)
        member = next(m for m in members if m.name == './runs')
        member.type = tarfile.REGTYPE
        row = next(r for r in inventory['members'] if r['path'] == './runs')
        row.update(type='file', tar_type='0', sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'file/directory conflict'):
            VERIFY.validate_members(members, inventory)

    def test_existing_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'must be fresh'):
                VERIFY.restore(BUNDLE, Path(folder).resolve())

    def test_destination_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            link = base / 'link'
            link.symlink_to(base / 'absent')
            with self.assertRaisesRegex(ValueError, 'must be fresh'):
                VERIFY.restore(BUNDLE, link)

    def test_symlink_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            link = base / 'link'
            link.symlink_to(base, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink aliases'):
                VERIFY.restore(BUNDLE, link / 'restore')

    def test_publication_mutations(self):
        for kind in ['tampered', 'symlink', 'extra', 'directory']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                base = Path(folder).resolve() / 'bundle'
                shutil.copytree(BUNDLE, base)
                if kind == 'tampered':
                    with (base / 'RESULTS.md').open('a') as stream:
                        stream.write('changed')
                elif kind == 'symlink':
                    (base / 'RESULTS.md').unlink()
                    (base / 'RESULTS.md').symlink_to(BUNDLE / 'RESULTS.md')
                elif kind == 'extra':
                    (base / 'unexpected.txt').touch()
                else:
                    (base / 'unexpected-directory').mkdir()
                output = base.parent / 'restore'
                with self.assertRaises(ValueError):
                    VERIFY.restore(base, output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
