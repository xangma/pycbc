"""Tests for the persistent inspiral session and campaign runner."""

import json
import os
import shutil
import tempfile
import unittest
import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc.filter.gpu_search.inspiral_session import (  # noqa: E402
    InspiralSession,
    _fingerprint_table,
)
from tools.run_inspiral_campaign import (  # noqa: E402
    run_campaign,
    _worker_cleanup,
)


def _create_mock_bank(path: str, num_templates: int = 5):
    with h5py.File(path, "w") as f:
        f.create_dataset("mass1", data=np.linspace(10.0, 15.0, num_templates))
        f.create_dataset("mass2", data=np.linspace(10.0, 15.0, num_templates))
        f.create_dataset("spin1z", data=np.zeros(num_templates))
        f.create_dataset("spin2z", data=np.zeros(num_templates))
        f.create_dataset("f_lower", data=np.full(num_templates, 30.0))
        f.attrs["approximant"] = "TaylorF2"


class TestInspiralSession(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.bank_path = os.path.join(self.tmpdir, "mock_bank.hdf")
        _create_mock_bank(self.bank_path)
        _worker_cleanup()

    def tearDown(self):
        _worker_cleanup()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fingerprint_table_deterministic(self):
        t1 = np.empty(2, dtype=[("name", object), ("mass", np.float32)])
        t1["name"] = np.array(["tmplA", "tmplB"], dtype=object)
        t1["mass"] = [10.0, 20.0]

        t2 = np.empty(2, dtype=[("mass", np.float32), ("name", object)])
        t2["name"] = np.array(["tmplA", "tmplB"], dtype=object)
        t2["mass"] = [10.0, 20.0]

        h1, n1 = _fingerprint_table(t1)
        h2, n2 = _fingerprint_table(t2)
        self.assertEqual(h1, h2)
        self.assertEqual(n1, n2)

        t3 = np.copy(t1)
        t3["name"][1] = "tmplC"
        h3, _ = _fingerprint_table(t3)
        self.assertNotEqual(h1, h3)

    def test_session_bank_binding_and_invalidation(self):
        from pycbc.waveform import FilterBank
        flen = 65
        df = 1.0
        bank = FilterBank(
            self.bank_path,
            flen,
            df,
            dtype=np.complex64,
            approximant="TaylorF2",
            low_frequency_cutoff=30.0,
        )
        session = InspiralSession(cache_bytes=1000000)

        cfg1 = {"approximant": "TaylorF2", "flow": 30.0}
        session.bind_bank(bank, cfg1)
        self.assertEqual(session.stats["bank_binds"], 1)
        self.assertEqual(session.stats["bank_hits"], 0)

        session.bind_bank(bank, cfg1)
        self.assertEqual(session.stats["bank_hits"], 1)

        cfg2 = {"approximant": "TaylorF2", "flow": 40.0}
        session.bind_bank(bank, cfg2)
        self.assertEqual(session.stats["bank_invalidations"], 1)

        f_handle = getattr(bank, "file", getattr(bank, "filehandler", None))
        if f_handle is not None:
            f_handle.close()

        with h5py.File(self.bank_path, "a") as f:
            f["mass1"][:] += 1.0

        bank_fresh = FilterBank(
            self.bank_path,
            flen,
            df,
            dtype=np.complex64,
            approximant="TaylorF2",
            low_frequency_cutoff=30.0,
        )
        session.bind_bank(bank_fresh, cfg2)
        self.assertEqual(session.stats["bank_invalidations"], 2)
        f_handle = getattr(
            bank_fresh, "file", getattr(bank_fresh, "filehandler", None)
        )
        if f_handle is not None:
            f_handle.close()

    def test_session_template_snapshot_isolation_and_sigmasq(self):
        from pycbc.waveform import FilterBank
        from pycbc.types import FrequencySeries

        flen = 129
        df = 1.0
        bank = FilterBank(
            self.bank_path,
            flen,
            df,
            dtype=np.complex64,
            approximant="TaylorF2",
            low_frequency_cutoff=30.0,
        )
        session = InspiralSession(cache_bytes=1000000)
        session.bind_bank(bank, {"approximant": "TaylorF2"})

        tmpl1 = session.template(bank, 0)
        self.assertEqual(session.stats["template_misses"], 1)
        self.assertEqual(session.stats["template_hits"], 0)

        tmpl2 = session.template(bank, 0)
        self.assertEqual(session.stats["template_hits"], 1)
        self.assertIsNot(tmpl1, tmpl2)
        np.testing.assert_array_equal(tmpl1.numpy(), tmpl2.numpy())

        tmpl2.data[10] += 999.0 + 999.0j
        tmpl3 = session.template(bank, 0)
        self.assertNotEqual(tmpl2.data[10], tmpl3.data[10])
        np.testing.assert_array_equal(tmpl1.numpy(), tmpl3.numpy())

        # Match the single-precision template PSD.
        psd_arr = np.linspace(1.0, 5.0, flen).astype(np.float32)
        psd_fs1 = FrequencySeries(psd_arr, delta_f=df)
        s1 = session.sigmasq(tmpl1, 0, psd_fs1)
        self.assertEqual(session.stats["sigmasq_misses"], 1)
        self.assertEqual(session.stats["sigmasq_hits"], 0)

        s2 = session.sigmasq(tmpl1, 0, psd_fs1)
        self.assertEqual(session.stats["sigmasq_hits"], 1)
        self.assertEqual(s1, s2)

        session.end_shard()
        psd_arr_mut = psd_arr.copy()
        psd_arr_mut[40] *= 10.0
        psd_fs2 = FrequencySeries(psd_arr_mut, delta_f=df)
        s3 = session.sigmasq(tmpl1, 0, psd_fs2)
        self.assertEqual(session.stats["sigmasq_misses"], 2)
        self.assertNotEqual(s1, s3)

        f_handle = getattr(bank, "file", getattr(bank, "filehandler", None))
        if f_handle is not None:
            f_handle.close()

    def test_eviction_and_zero_budget(self):
        from pycbc.waveform import FilterBank
        flen = 129
        df = 1.0
        bank = FilterBank(
            self.bank_path,
            flen,
            df,
            dtype=np.complex64,
            approximant="TaylorF2",
            low_frequency_cutoff=30.0,
        )

        session_zero = InspiralSession(cache_bytes=0)
        session_zero.bind_bank(bank, {})
        _ = session_zero.template(bank, 0)
        _ = session_zero.template(bank, 0)
        self.assertEqual(session_zero.stats["template_misses"], 2)
        self.assertEqual(session_zero.stats["template_hits"], 0)
        self.assertEqual(session_zero.current_bytes, 0)

        # Budget permitting only ~1 template plus metadata
        tmpl0 = bank[0]
        entry_bytes = tmpl0.nbytes + 2048
        session_small = InspiralSession(cache_bytes=int(entry_bytes * 1.5))
        session_small.bind_bank(bank, {})
        _ = session_small.template(bank, 0)
        self.assertEqual(session_small.stats["template_misses"], 1)
        _ = session_small.template(bank, 1)
        self.assertEqual(session_small.stats["template_misses"], 2)
        self.assertGreater(session_small.stats["evictions"], 0)

        f_handle = getattr(bank, "file", getattr(bank, "filehandler", None))
        if f_handle is not None:
            f_handle.close()

    def test_campaign_validation_and_runner(self):
        m_path = os.path.join(self.tmpdir, "manifest.json")
        out_dir = os.path.join(self.tmpdir, "out_campaign")

        bad_manifest = {
            "shards": [
                {"argv": ["--batch-size", "4", "--output", "foo.hdf"]}
            ]
        }
        with open(m_path, "w") as f:
            json.dump(bad_manifest, f)

        with self.assertRaises(ValueError):
            run_campaign(m_path, out_dir)
        self.assertFalse(os.path.exists(out_dir))

        mock_script = os.path.join(self.tmpdir, "mock_inspiral.py")
        with open(mock_script, "w") as f:
            f.write(
                "import sys, h5py, numpy as np\n"
                "out = sys.argv[sys.argv.index('--output') + 1]\n"
                "with h5py.File(out, 'w') as h:\n"
                "    h.create_dataset('status', data=np.array([1]))\n"
            )

        valid_manifest = {
            "shards": [
                {"argv": ["--batch-size", "4", "--bank-file", self.bank_path]},
                {"argv": ["--batch-size", "4", "--bank-file", self.bank_path]},
            ]
        }
        with open(m_path, "w") as f:
            json.dump(valid_manifest, f)

        receipt = run_campaign(
            m_path, out_dir, fresh_workers=False,
            inspiral_entrypoint=mock_script
        )
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(len(receipt["shards"]), 2)
        pids = [s["pid"] for s in receipt["shards"]]
        self.assertEqual(len(set(pids)), 1)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "receipt.json")))
        fresh = run_campaign(
            m_path, out_dir + "_fresh", fresh_workers=True,
            inspiral_entrypoint=mock_script)
        self.assertEqual(len({s["pid"] for s in fresh["shards"]}), 2)
        for cached, uncached in zip(receipt["shards"], fresh["shards"]):
            with h5py.File(cached["output_file"]) as a, h5py.File(
                    uncached["output_file"]) as b:
                np.testing.assert_array_equal(a["status"][:], b["status"][:])

        with self.assertRaises(FileExistsError):
            run_campaign(m_path, out_dir, inspiral_entrypoint=mock_script)

    def test_campaign_failure_and_partial_receipt(self):
        m_path = os.path.join(self.tmpdir, "manifest_fail.json")
        out_dir = os.path.join(self.tmpdir, "out_fail")
        fail_script = os.path.join(self.tmpdir, "mock_fail.py")
        with open(fail_script, "w") as f:
            f.write(
                "import sys, h5py, numpy as np\n"
                "if '--fail' in sys.argv:\n"
                "    sys.exit(42)\n"
                "out = sys.argv[sys.argv.index('--output') + 1]\n"
                "with h5py.File(out, 'w') as h:\n"
                "    h.create_dataset('status', data=np.array([1]))\n"
            )

        fail_manifest = {
            "shards": [
                {"argv": ["--batch-size", "4", "--bank-file", self.bank_path]},
                {"argv": ["--batch-size", "4", "--bank-file",
                          self.bank_path, "--fail"]},
            ]
        }
        with open(m_path, "w") as f:
            json.dump(fail_manifest, f)

        with self.assertRaises(RuntimeError):
            run_campaign(
                m_path, out_dir, fresh_workers=False,
                inspiral_entrypoint=fail_script
            )

        receipt_file = os.path.join(out_dir, "receipt.json")
        self.assertTrue(os.path.exists(receipt_file))
        with open(receipt_file, "r") as f:
            receipt = json.load(f)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["failure_index"], 1)
        self.assertEqual(len(receipt["shards"]), 1)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize('device', ['cpu'] + (
        ['cuda'] if torch.cuda.is_available() else []))
def test_cross_shard_template_metadata_psd_and_table_invalidation(
        tmp_path, device):
    from pycbc.scheme import TorchScheme
    from pycbc.types import FrequencySeries
    from pycbc.waveform import FilterBank

    bank_path = str(tmp_path / 'bank.hdf')
    _create_mock_bank(bank_path)
    session = InspiralSession(1000000)
    with TorchScheme(device=device):
        def bank():
            return FilterBank(bank_path, 129, 1., dtype=np.complex64,
                              approximant='TaylorF2',
                              low_frequency_cutoff=30.)

        first = bank()
        session.bind_bank(first, {'device': device})
        generated = session.template(first, 0)
        psd = FrequencySeries(np.ones(129, dtype=np.float32), delta_f=1.)
        sigma = session.sigmasq(generated, 0, psd)
        first.file.close()
        session.end_shard()
        second = bank()
        session.bind_bank(second, {'device': device})
        cached = session.template(second, 0)
        fresh = second[0]
        assert session.stats['bank_hits'] == 1
        assert session.stats['template_hits'] == 1
        assert cached.sigmasq.__self__ is cached
        for attr in ('f_lower', 'min_f_lower', 'end_idx', 'chirp_length',
                     'length_in_time', 'approximant', 'end_frequency'):
            assert getattr(cached, attr) == getattr(fresh, attr)
        np.testing.assert_array_equal(cached.numpy(), fresh.numpy())
        np.testing.assert_array_equal(cached.params, fresh.params)
        assert session.sigmasq(cached, 0, psd.copy()) == sigma
        assert session.stats['sigmasq_hits'] == 1
        session.end_shard()
        psd = FrequencySeries(np.full(129, 2, dtype=np.float32),
                              delta_f=1.)
        changed_sigma = session.sigmasq(cached, 0, psd)
        assert changed_sigma != sigma
        assert changed_sigma == fresh.sigmasq(psd)
        second.table = second.table[1:]
        session.bind_bank(second, {'device': device})
        assert session.stats['bank_invalidations'] == 1
        assert session.current_bytes == 0
        second.file.close()
        session.close()


def test_worker_failure_restores_argv_and_clears_cache(tmp_path):
    import sys
    from tools import run_inspiral_campaign as runner

    fail_script = tmp_path / 'fail.py'
    fail_script.write_text('raise RuntimeError("intentional failure")\n')
    argv = sys.argv
    runner._WORKER_SESSION = InspiralSession(1024)
    with pytest.raises(RuntimeError, match='intentional failure'):
        runner.execute_inspiral_shard({
            'argv': ['--batch-size', '4'],
            'output_file': str(tmp_path / 'out.hdf'),
            'inspiral_entrypoint': str(fail_script), 'cache_bytes': 1024,
        })
    assert sys.argv is argv
    assert runner._WORKER_SESSION is None


def test_scalar_cache_eviction(tmp_path):
    from types import SimpleNamespace
    from pycbc.types import FrequencySeries

    session = InspiralSession(1024)
    template = SimpleNamespace(sigmasq=lambda psd: float(psd.numpy().sum()))
    first = FrequencySeries(np.ones(8, dtype=np.float32), delta_f=1.)
    second = FrequencySeries(np.full(8, 2, dtype=np.float32), delta_f=1.)
    assert session.sigmasq(template, 0, first) == 8.
    assert session.sigmasq(template, 0, second) == 16.
    assert session.stats['evictions'] == 1
    assert session.current_bytes == 1024
    assert session.sigmasq(template, 0, first) == 8.
    assert session.stats['sigmasq_hits'] == 0


@pytest.mark.parametrize('extra', [
    ['--checkpoint-interval', '60'], ['--require-valid-checkpoint'],
    ['--checkpoint-exit-maxtime=100'], ['--multiprocessing-nprocesses', '2'],
    ['--output=unmanaged.hdf'], ['--help'], ['--version'],
])
def test_forbidden_options_rejected_before_output_creation(tmp_path, extra):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'shards': [{'argv': ['--batch-size', '4'] + extra}]}))
    output = tmp_path / 'output'
    with pytest.raises(ValueError):
        run_campaign(str(manifest), str(output))
    assert not output.exists()
