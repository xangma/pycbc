"""Small, isolated transport fixtures; no campaign, Git, network or real archive writes."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import archive_transport as transport


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pycbc-transport-fixture-')
        self.root = Path(self.temp.name)
        self.original = self.root / 'original'
        self.original.mkdir()
        self.files = {}
        self.add('runs/native/perf.data', b'original sample bytes\x00' * 2048, 0o640)
        self.add('report.json', b'{"scientific_hashes":"unchanged"}\n')
        self.before = copy.deepcopy(self.files)

    def tearDown(self):
        for item in self.before.values():
            path = Path(item['source'])
            if path.exists():
                # Explicit changed-source test restores its input before returning.
                self.assertEqual(transport.file_record(path), transport.metadata(item))
        self.temp.cleanup()

    def add(self, name, data, mode=0o644):
        path = self.original / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
        self.files[name] = dict(source=str(path), **transport.file_record(path))

    def prepare(self, name='cache', **kwargs):
        return transport.prepare_transport(self.files, self.root / name, threshold=1024, **kwargs)

    def archive(self):
        actual, info = self.prepare()
        archive = self.root / 'archive'
        archive.mkdir()
        for name, item in actual.items():
            path = archive / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item['source'], path)
            path.chmod(item['mode'])
        return archive, actual, info

    def mutate_manifest(self, archive, mutate):
        path = archive / transport.MANIFEST
        value = json.loads(path.read_text())
        mutate(value)
        path.write_bytes(transport.canonical(value))

    def test_full_roundtrip_and_separate_inventories(self):
        archive, actual, info = self.archive()
        self.assertNotIn('runs/native/perf.data', actual)
        self.assertEqual(info['compressed_files'], 1)
        manifest = transport.verify_archive(archive)
        self.assertEqual(manifest['logical_files'], {k: transport.metadata(v) for k, v in self.files.items()})
        archived_before = {str(p.relative_to(archive)): transport.file_record(p)
                           for p in archive.rglob('*') if p.is_file()}
        destination = self.root / 'restored'
        self.assertEqual(transport.restore_archive(archive, destination), len(self.files))
        self.assertEqual({str(p.relative_to(destination)): transport.file_record(p)
                          for p in destination.rglob('*') if p.is_file()}, manifest['logical_files'])
        self.assertEqual({str(p.relative_to(archive)): transport.file_record(p)
                          for p in archive.rglob('*') if p.is_file()}, archived_before)

    def test_compression_determinism_and_frozen_reuse(self):
        first, _ = self.prepare()
        second, _ = self.prepare('cache-second')
        self.assertEqual({k: transport.metadata(v) for k, v in first.items()},
                         {k: transport.metadata(v) for k, v in second.items()})
        mtimes = {v['source']: Path(v['source']).stat().st_mtime_ns for v in first.values()}
        self.assertEqual(self.prepare(create=False)[0], first)
        self.assertEqual({p: Path(p).stat().st_mtime_ns for p in mtimes}, mtimes)

    def test_no_large_files_keep_original_inventory(self):
        small = {'report.json': self.files['report.json']}
        actual, info = transport.prepare_transport(small, self.root / 'unused', threshold=1024)
        self.assertEqual(actual, small)
        self.assertIsNone(info)
        self.assertFalse((self.root / 'unused').exists())

    def test_exact_threshold_remains_uncompressed(self):
        self.add('boundary.dat', b'a' * 1024)
        self.add('above.dat', b'b' * 1025)
        actual, info = self.prepare()
        self.assertIn('boundary.dat', actual)
        self.assertNotIn('above.dat', actual)
        self.assertEqual(info['compressed_files'], 2)

    def test_identical_inputs_share_gzip_preserve_modes(self):
        self.add('other/perf.data', Path(self.files['runs/native/perf.data']['source']).read_bytes(), 0o755)
        archive, actual, info = self.archive()
        self.assertEqual(info['compressed_files'], 2)
        self.assertEqual(sum(name.endswith('.gz') for name in actual), 1)
        destination = self.root / 'restored'
        transport.restore_archive(archive, destination)
        self.assertEqual(transport.file_record(destination / 'other/perf.data')['mode'], 0o755)

    def test_execute_requires_frozen_cache(self):
        with self.assertRaisesRegex(ValueError, 'Missing frozen transport cache'):
            self.prepare(create=False)
        self.assertFalse((self.root / 'cache').exists())

    def test_original_change_rejected(self):
        path = Path(self.files['runs/native/perf.data']['source'])
        original = path.read_bytes()
        try:
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Original evidence changed'):
                self.prepare()
        finally:
            path.write_bytes(original)

    def test_changed_cached_gzip_rejected(self):
        actual, _ = self.prepare()
        path = Path(next(item['source'] for name, item in actual.items() if name.endswith('.gz')))
        path.write_bytes(b'corrupted cache')
        with self.assertRaises(ValueError):
            self.prepare(create=False)

    def test_changed_cached_manifest_rejected(self):
        actual, _ = self.prepare()
        Path(actual[transport.MANIFEST]['source']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Frozen transport file changed'):
            self.prepare(create=False)

    def test_incompressible_file_rejected_before_archive(self):
        self.add('random.bin', random.Random(943).randbytes(4096))
        with self.assertRaisesRegex(ValueError, 'Compressed file still exceeds archive limit'):
            self.prepare(maximum=2048)
        self.assertFalse((self.root / 'archive').exists())

    def test_unsafe_paths_and_reserved_prefix(self):
        for name in ('../outside', '/absolute', 'a//b', 'a/./b', 'a\\b', 'a\nb',
                     'archive-transport/fake', 'SHA256SUMS'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                transport.prepare_transport({name: self.files['report.json']}, self.root / 'cache', threshold=1024)

    def test_policy_cannot_raise_production_limits(self):
        for kwargs in ({'threshold': transport.THRESHOLD + 1}, {'threshold': 0},
                       {'threshold': 1024, 'maximum': transport.MAXIMUM + 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                transport.prepare_transport(self.files, self.root / 'cache', **kwargs)

    def test_symlink_cache_rejected(self):
        (self.root / 'cache').symlink_to(self.original, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink transport cache'):
            self.prepare()

    def test_corrupted_archived_gzip_rejected(self):
        archive, actual, _ = self.archive()
        path = archive / next(name for name in actual if name.endswith('.gz'))
        path.write_bytes(path.read_bytes()[:-7])
        with self.assertRaisesRegex(ValueError, 'Archive file changed'):
            transport.verify_archive(archive)

    def test_corrupt_gzip_rebound_hash_still_rejected(self):
        archive, actual, _ = self.archive()
        path = archive / next(name for name in actual if name.endswith('.gz'))
        path.write_bytes(path.read_bytes()[:-7])
        self.mutate_manifest(archive, lambda m: m['compressed_files']['runs/native/perf.data'].update(
            transport.file_record(path)))
        with self.assertRaises((ValueError, EOFError)):
            transport.verify_archive(archive)

    def test_changed_uncompressed_file_rejected(self):
        archive, _, _ = self.archive()
        (archive / 'report.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Archive file changed'):
            transport.verify_archive(archive)

    def test_missing_mapping_rejected(self):
        archive, _, _ = self.archive()
        self.mutate_manifest(archive, lambda m: m['compressed_files'].clear())
        with self.assertRaisesRegex(ValueError, 'Incomplete compressed-file mapping'):
            transport.verify_archive(archive)

    def test_wrong_decompressed_size_rejected(self):
        archive, _, _ = self.archive()
        self.mutate_manifest(archive, lambda m: m['logical_files']['runs/native/perf.data'].update(bytes=2048))
        with self.assertRaisesRegex(ValueError, 'Expanded file exceeds recorded size'):
            transport.verify_archive(archive)

    def test_wrong_decompressed_hash_rejected(self):
        actual, _ = self.prepare()
        path = Path(next(item['source'] for name, item in actual.items() if name.endswith('.gz')))
        expected = dict(self.files['runs/native/perf.data'], sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'Reconstructed bytes differ'):
            transport.verify_gzip(path, expected)

    def test_manifest_traversal_rejected(self):
        archive, _, _ = self.archive()
        def mutate(m):
            m['compressed_files']['runs/native/perf.data']['path'] = '../outside.gz'
        self.mutate_manifest(archive, mutate)
        with self.assertRaisesRegex(ValueError, 'Invalid compressed-file path'):
            transport.verify_archive(archive)

    def test_symlink_archived_input_rejected(self):
        archive, _, _ = self.archive()
        path = archive / 'report.json'
        path.unlink()
        path.symlink_to(self.original / 'report.json')
        with self.assertRaisesRegex(ValueError, 'Symlink in archive path'):
            transport.verify_archive(archive)

    def test_restore_refuses_existing_or_in_archive_destination(self):
        archive, _, _ = self.archive()
        for destination in (self.original, archive / 'restored'):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                transport.restore_archive(archive, destination)

    def test_cli_is_standalone_in_extracted_supplement(self):
        archive, _, _ = self.archive()
        helper = archive / transport.RESTORE
        result = subprocess.run([sys.executable, '-I', '-B', str(helper), '--verify-only'],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'verified')
        result = subprocess.run([sys.executable, '-I', '-B', str(helper), '--destination', str(self.root / 'restored')],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'restored')

    def stage_fixture(self):
        stage_path = Path(__file__).with_name('stage-evidence.py')
        spec = importlib.util.spec_from_file_location('isolated_stage_evidence', stage_path)
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        stage.HERE = self.root / 'publication'
        stage.HERE.mkdir()
        shutil.copyfile(transport.__file__, stage.HERE / 'archive_transport.py')
        art, reference, archive = (self.root / name for name in ('art', 'reference', 'archive'))
        for path in (art, reference, archive):
            path.mkdir()
        (art / 'receipt.json').write_text('{"supporting":true}\n')
        (reference / 'perf.data').write_bytes(b'large fixture evidence\x00' * 2048)
        (reference / 'report.json').write_text('{"status":"pass","fixture":true}\n')
        (archive / 'historical.txt').write_text('immutable historical fixture\n')
        def inventory(root):
            return {p.name: dict(source=str(p), **transport.file_record(p)) for p in sorted(root.iterdir())}
        observed = []
        stage.inventory = inventory
        stage.reference_inventory = lambda root: (inventory(root), {})
        stage.qualify = lambda root, files: observed.append(('supporting', set(files)))
        def qualify(root, files, supporting):
            observed.append(('reference', set(files)))
            self.assertIn('perf.data', files)
            self.assertFalse(any(name.endswith('.gz') for name in files))
            return {'isolated_science_fixture': True}
        stage.qualify_reference = qualify
        def qualify_hotpath(root, files):
            observed.append(('optimized', set(files)))
            self.assertIn('perf.data', files)
            self.assertFalse(any(name.endswith('.gz') for name in files))
            return {'isolated_optimized_science_fixture': True}
        stage.qualify_hotpath_reference = qualify_hotpath
        args = [str(stage_path), '--art', str(art), '--reference', str(reference),
                '--archive', str(archive), '--transport-threshold-bytes', '1024']
        def run(execute=False):
            with patch.object(sys, 'argv', args + (['--execute'] if execute else [])), redirect_stdout(io.StringIO()):
                stage.main()
        return stage, art, reference, archive, observed, run

    def test_stage_plan_and_execute_use_distinct_inventories(self):
        stage, art, reference, archive, observed, run = self.stage_fixture()
        run()
        plan = json.loads((stage.HERE / 'evidence-staging-plan.json').read_text())
        self.assertEqual(plan['schema'], 'torch-reference-and-performance-fix-stage-v6')
        supplement = plan['supplements'][stage.DESTINATIONS[1]]
        self.assertIn('perf.data', supplement['logical_files'])
        self.assertNotIn('perf.data', supplement['files'])
        self.assertEqual(supplement['optimized_qualification'], {'isolated_optimized_science_fixture': True})
        self.assertEqual(set(p.name for p in archive.iterdir()), {'historical.txt'})
        cache_before = {str(p): transport.file_record(p) for p in (stage.HERE / 'transport-cache').rglob('*') if p.is_file()}
        run(execute=True)
        self.assertEqual({str(p): transport.file_record(p) for p in (stage.HERE / 'transport-cache').rglob('*') if p.is_file()}, cache_before)
        self.assertEqual((archive / 'historical.txt').read_text(), 'immutable historical fixture\n')
        extracted = archive / stage.DESTINATIONS[1]
        transport.verify_archive(extracted)
        checksums = (extracted / 'SHA256SUMS').read_text().splitlines()
        self.assertEqual({line.split('  ', 1)[1] for line in checksums}, set(supplement['files']))
        self.assertEqual(observed, [('supporting', {'receipt.json'}), ('reference', {'perf.data', 'report.json'}),
                                    ('optimized', {'perf.data', 'report.json'})] * 2)

    def test_stage_corrupted_cache_aborts_before_archive_copy(self):
        stage, _, _, archive, _, run = self.stage_fixture()
        run()
        next((stage.HERE / 'transport-cache').rglob('*.gz')).write_bytes(b'corrupted')
        with self.assertRaises(ValueError):
            run(execute=True)
        self.assertEqual(set(p.name for p in archive.iterdir()), {'historical.txt'})

    def test_stage_old_or_changed_plan_aborts_before_copy(self):
        stage, _, _, archive, _, run = self.stage_fixture()
        run()
        path = stage.HERE / 'evidence-staging-plan.json'
        plan = json.loads(path.read_text())
        plan['schema'] = 'torch-reference-and-performance-fix-stage-v5'
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'Frozen plan differs'):
            run(execute=True)
        self.assertEqual(set(p.name for p in archive.iterdir()), {'historical.txt'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
