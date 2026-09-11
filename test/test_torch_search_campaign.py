"""Unit tests for persistent SearchEngine campaign execution."""

import json
import os
import shutil
import tempfile
import unittest
import numpy as np
import pytest

from tools.run_torch_search_campaign import (
    execute_shard,
    run_campaign,
    _close_state,
)


pytest.importorskip("torch")


def _make_npz_files(tmpdir: str):
    flen = 129
    delta_f = np.float64(1.0)
    b = 7
    rng = np.random.RandomState(42)

    tmps1 = (rng.randn(b, flen) + 1j * rng.randn(b, flen)).astype(np.complex64)
    t_ids1 = np.arange(100, 100 + b, dtype=np.int64)
    bank1 = os.path.join(tmpdir, "bank1.npz")
    np.savez(bank1, templates=tmps1, template_ids=t_ids1, delta_f=delta_f)

    tmps2 = (rng.randn(b, flen) + 1j * rng.randn(b, flen)).astype(np.complex64)
    t_ids2 = np.arange(200, 200 + b, dtype=np.int64)
    bank2 = os.path.join(tmpdir, "bank2.npz")
    np.savez(bank2, templates=tmps2, template_ids=t_ids2, delta_f=delta_f)

    psd1_arr = np.linspace(1.0, 5.0, flen).astype(np.float64)
    psd1 = os.path.join(tmpdir, "psd1.npz")
    np.savez(psd1, psd=psd1_arr, delta_f=delta_f)

    psd2_arr = np.linspace(2.0, 8.0, flen).astype(np.float64)
    psd2 = os.path.join(tmpdir, "psd2.npz")
    np.savez(psd2, psd=psd2_arr, delta_f=delta_f)

    # Injected signals to guarantee non-empty candidate triggers
    s1 = tmps1[0].copy() * 100.0
    strain1 = os.path.join(tmpdir, "strain1.npz")
    np.savez(strain1, strain=s1, delta_f=delta_f)

    s2 = tmps1[1].copy() * 100.0
    strain2 = os.path.join(tmpdir, "strain2.npz")
    np.savez(strain2, strain=s2, delta_f=delta_f)

    return bank1, bank2, psd1, psd2, strain1, strain2


class TestSearchCampaign(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _close_state()

    def tearDown(self):
        _close_state()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_campaign_exact_parity_fresh_vs_persistent(self):
        b1, b2, p1, p2, s1, s2 = _make_npz_files(self.tmpdir)
        shards = [
            {
                "bank": b1,
                "psd": p1,
                "strain": s1,
                "block_id": 0,
                "valid_interval": [32, 96],
            },
            {
                "bank": b1,
                "psd": p1,
                "strain": s2,
                "block_id": 1,
                "valid_interval": [32, 96],
            },
            {
                "bank": b1,
                "psd": p2,
                "strain": s1,
                "block_id": 2,
                "valid_interval": [32, 96],
            },
            {
                "bank": b2,
                "psd": p1,
                "strain": s1,
                "block_id": 3,
                "valid_interval": [32, 96],
            },
        ]
        m_path = os.path.join(self.tmpdir, "manifest.json")
        with open(m_path, "w") as f:
            json.dump({"shards": shards}, f)

        out_pers = os.path.join(self.tmpdir, "out_pers")
        rec_pers = run_campaign(
            m_path, out_pers, device="cpu", tile_size=3, fresh_workers=False
        )
        tasks_p = rec_pers["tasks"]
        self.assertEqual(len(set(t["pid"] for t in tasks_p)), 1)
        self.assertFalse(tasks_p[0]["cache_hit_bank"])
        self.assertTrue(tasks_p[1]["cache_hit_bank"])
        self.assertTrue(tasks_p[1]["cache_hit_psd"])
        self.assertTrue(tasks_p[2]["cache_hit_bank"])
        self.assertFalse(tasks_p[2]["cache_hit_psd"])
        self.assertFalse(tasks_p[3]["cache_hit_bank"])

        out_fresh = os.path.join(self.tmpdir, "out_fresh")
        rec_fresh = run_campaign(
            m_path, out_fresh, device="cpu", tile_size=3, fresh_workers=True
        )
        tasks_f = rec_fresh["tasks"]
        self.assertEqual(len(set(t["pid"] for t in tasks_f)), 4)
        for t in tasks_f:
            self.assertFalse(t["cache_hit_bank"])
            self.assertFalse(t["cache_hit_psd"])

        # Verify nonempty triggers and exact dtype, shape, and values.
        fields = [
            "template_id",
            "template_idx",
            "sample_idx",
            "snr",
            "sigmasq",
        ]
        expected_dtypes = {
            "template_id": np.dtype("int64"),
            "template_idx": np.dtype("int64"),
            "sample_idx": np.dtype("int64"),
            "snr": np.dtype("complex64"),
        }
        total_triggers = 0
        for b_id in range(4):
            file_p = os.path.join(out_pers, f"block_{b_id}.npz")
            file_f = os.path.join(out_fresh, f"block_{b_id}.npz")
            with np.load(file_p) as npz_p, np.load(file_f) as npz_f:
                self.assertEqual(npz_p["block_id"], npz_f["block_id"])
                for tile_idx in range(3):
                    for fld in fields:
                        k = f"tile{tile_idx}_{fld}"
                        arr_p, arr_f = npz_p[k], npz_f[k]
                        self.assertEqual(arr_p.dtype, arr_f.dtype)
                        if fld in expected_dtypes:
                            self.assertEqual(arr_p.dtype, expected_dtypes[fld])
                        self.assertEqual(arr_p.shape, arr_f.shape)
                        np.testing.assert_array_equal(arr_p, arr_f)
                        if fld == "snr":
                            total_triggers += len(arr_p)
        self.assertGreater(
            total_triggers,
            0,
            "Expected non-empty candidate trigger detections",
        )

    def test_empty_tiles_retain_bank_indices(self):
        b1, _, p1, _, s1, _ = _make_npz_files(self.tmpdir)
        with np.load(b1) as data:
            tmps, ids, df = (
                data["templates"],
                data["template_ids"],
                data["delta_f"],
            )
        tmps[:6] = 0
        np.savez(b1, templates=tmps, template_ids=ids, delta_f=df)
        spec = dict(
            bank=b1,
            psd=p1,
            strain=s1,
            block_id=9,
            valid_interval=[32, 96],
            device="cpu",
            tile_size=3,
            snr_threshold=5.5,
            cluster_window=64,
            f_lower=20.0,
        )
        result = execute_shard(spec)
        self.assertEqual(len(result["tile_results"]), 3)
        self.assertEqual(len(result["tile_results"][0]["snr"]), 0)
        self.assertEqual(len(result["tile_results"][1]["snr"]), 0)
        self.assertGreater(len(result["tile_results"][2]["snr"]), 0)
        np.testing.assert_array_equal(
            np.unique(result["tile_results"][2]["template_id"]), ids[6:]
        )
        np.savez(s1, strain=np.zeros(129, dtype=np.complex64), delta_f=df)
        empty = execute_shard(spec)
        self.assertTrue(empty["cache_hit_bank"])
        self.assertEqual(len(empty["tile_results"]), 3)
        self.assertTrue(
            all(len(tile["snr"]) == 0 for tile in empty["tile_results"])
        )

    def test_output_overwrite_error(self):
        b1, _, p1, _, s1, _ = _make_npz_files(self.tmpdir)
        shards = [
            {
                "bank": b1,
                "psd": p1,
                "strain": s1,
                "block_id": 0,
                "valid_interval": [32, 96],
            }
        ]
        m_path = os.path.join(self.tmpdir, "manifest.json")
        with open(m_path, "w") as f:
            json.dump({"shards": shards}, f)

        out_dir = os.path.join(self.tmpdir, "out_exists")
        os.makedirs(out_dir)
        with self.assertRaises(FileExistsError):
            run_campaign(m_path, out_dir, device="cpu")

    def test_inplace_bank_and_id_mutation_invalidates(self):
        b1, _, p1, _, s1, _ = _make_npz_files(self.tmpdir)
        spec = {
            "bank": b1,
            "psd": p1,
            "strain": s1,
            "block_id": 10,
            "valid_interval": [32, 96],
            "device": "cpu",
            "tile_size": 3,
            "snr_threshold": 5.5,
            "cluster_window": 64,
            "f_lower": 20.0,
        }
        res1 = execute_shard(spec)
        self.assertFalse(res1["cache_hit_bank"])

        # 1. In-place bank templates mutation
        with np.load(b1) as data:
            tmps, t_ids, df = (
                data["templates"],
                data["template_ids"],
                data["delta_f"],
            )
        np.savez(b1, templates=tmps * 1.5, template_ids=t_ids, delta_f=df)
        res2 = execute_shard(spec)
        self.assertFalse(
            res2["cache_hit_bank"],
            "Bank templates modification must miss cache",
        )

        # 2. In-place template_ids mutation
        with np.load(b1) as data:
            tmps, t_ids, df = (
                data["templates"],
                data["template_ids"],
                data["delta_f"],
            )
        np.savez(b1, templates=tmps, template_ids=t_ids + 1000, delta_f=df)
        res3 = execute_shard(spec)
        self.assertFalse(
            res3["cache_hit_bank"], "Template IDs modification must miss cache"
        )

    def test_inplace_psd_mutation_and_changed_policy_reuse(self):
        b1, _, p1, _, s1, _ = _make_npz_files(self.tmpdir)
        spec = {
            "bank": b1,
            "psd": p1,
            "strain": s1,
            "block_id": 10,
            "valid_interval": [32, 96],
            "device": "cpu",
            "tile_size": 3,
            "snr_threshold": 5.5,
            "cluster_window": 64,
            "f_lower": 20.0,
        }
        res1 = execute_shard(spec)
        self.assertFalse(res1["cache_hit_psd"])

        # In-place PSD mutation
        with np.load(p1) as data:
            psd_arr, df = data["psd"], data["delta_f"]
        np.savez(p1, psd=psd_arr * 2.0, delta_f=df)
        res2 = execute_shard(spec)
        self.assertTrue(res2["cache_hit_bank"])
        self.assertFalse(
            res2["cache_hit_psd"], "PSD mutation must miss PSD cache"
        )

        # Policy change (snr_threshold) must invalidate engine
        spec_new_pol = dict(spec)
        spec_new_pol["snr_threshold"] = 4.0
        res3 = execute_shard(spec_new_pol)
        self.assertFalse(
            res3["cache_hit_bank"],
            "Policy change must invalidate bank and engine",
        )

    def test_malformed_fixtures_and_cache_recovery(self):
        b1, _, p1, _, s1, _ = _make_npz_files(self.tmpdir)
        spec = {
            "bank": b1,
            "psd": p1,
            "strain": s1,
            "block_id": 20,
            "valid_interval": [32, 96],
            "device": "cpu",
            "tile_size": 3,
            "snr_threshold": 5.5,
            "cluster_window": 64,
            "f_lower": 20.0,
        }
        res1 = execute_shard(spec)
        self.assertFalse(res1["cache_hit_bank"])

        # 1. Invalid f_lower >= Nyquist
        spec_bad_freq = dict(spec)
        spec_bad_freq["f_lower"] = 200.0  # Nyquist is 128.0
        with self.assertRaises(ValueError):
            execute_shard(spec_bad_freq)

        # 2. Corrupted NaN strain
        bad_strain = os.path.join(self.tmpdir, "bad_strain.npz")
        nan_arr = np.full(129, np.nan, dtype=np.complex64)
        np.savez(bad_strain, strain=nan_arr, delta_f=np.float64(1.0))
        spec_bad_strain = dict(spec)
        spec_bad_strain["strain"] = bad_strain
        with self.assertRaises(ValueError):
            execute_shard(spec_bad_strain)

        # Recovery execution succeeds cold
        res_rec = execute_shard(spec)
        self.assertFalse(res_rec["cache_hit_bank"])
        self.assertFalse(res_rec["cache_hit_psd"])


if __name__ == "__main__":
    unittest.main()
