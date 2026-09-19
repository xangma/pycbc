"""Tests for the JAX multi-arm inspiral benchmark campaign runner."""

import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np

from tools.bench_jax_inspiral_campaign import (
    compare_trigger_parity,
    sample_summary,
    _parse_stderr_phases,
    ARM_NAMES,
    DEFAULT_ARMS,
    BATCHED_ARMS,
)


class TestBenchJaxInspiralCampaign(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_sample_summary(self):
        samples = [10.0, 12.0, 11.0, 9.0, 13.0]
        res = sample_summary(samples, "seconds")
        self.assertEqual(res["count"], 5)
        self.assertAlmostEqual(res["median"], 11.0)
        self.assertAlmostEqual(res["mean"], 11.0)
        self.assertEqual(res["min"], 9.0)
        self.assertEqual(res["max"], 13.0)
        self.assertTrue("median_ci95" in res)
        self.assertLessEqual(res["median_ci95"]["low"], res["median"])
        self.assertGreaterEqual(res["median_ci95"]["high"], res["median"])

    def test_sample_summary_empty(self):
        res = sample_summary([], "seconds")
        self.assertEqual(res["count"], 0)
        self.assertEqual(res["samples"], [])

    def test_parse_stderr_phases_fallback(self):
        phases = _parse_stderr_phases([], process_wall_sec=100.0, calc_time_sec=60.0, tsetup_sec=15.0)
        self.assertEqual(phases["matched_filter_sec"], 60.0)
        self.assertEqual(phases["conditioning_sec"], 15.0)
        self.assertEqual(phases["startup_import_sec"], 25.0)

    def test_parse_stderr_phases_parsed(self):
        log_lines = [
            "2026-09-18T10:00:00.000 Reading Frames...",
            "2026-09-18T10:00:10.000 Read in template bank...",
            "2026-09-18T10:00:12.000 Filtering template 0...",
            "2026-09-18T10:00:50.000 We currently have 5 triggers...",
            "2026-09-18T10:00:52.000 Writing out triggers...",
            "2026-09-18T10:00:55.000 Finished",
        ]
        phases = _parse_stderr_phases(log_lines, process_wall_sec=60.0, calc_time_sec=30.0, tsetup_sec=10.0)
        self.assertGreater(phases["conditioning_sec"], 0.0)
        self.assertEqual(phases["matched_filter_sec"], 30.0)
        self.assertGreater(phases["waveform_prep_sec"], 0.0)
        self.assertGreaterEqual(phases["startup_import_sec"], 0.0)

    def test_compare_trigger_parity_exact(self):
        f1 = self.path / "base.hdf"
        f2 = self.path / "cand.hdf"

        with h5py.File(f1, "w") as fb, h5py.File(f2, "w") as fc:
            snrs = np.array([8.0, 9.5, 12.0, 15.5])
            times = np.array([100.1, 105.2, 110.3, 115.4])
            fb.create_dataset("H1/snr", data=snrs)
            fb.create_dataset("H1/end_time", data=times)
            fc.create_dataset("H1/snr", data=snrs)
            fc.create_dataset("H1/end_time", data=times)

        res = compare_trigger_parity(f1, f2)
        self.assertTrue(res["passed"])
        self.assertEqual(res["trigger_count"], 4)
        self.assertEqual(res["max_snr_diff"], 0.0)
        self.assertEqual(res["relative_l2_snr"], 0.0)

    def test_compare_trigger_parity_tolerance(self):
        f1 = self.path / "base_tol.hdf"
        f2 = self.path / "cand_tol.hdf"

        with h5py.File(f1, "w") as fb, h5py.File(f2, "w") as fc:
            snrs_b = np.array([8.0, 10.0, 12.0])
            snrs_c = np.array([8.00001, 10.00001, 12.00001])
            times = np.array([100.0, 105.0, 110.0])
            fb.create_dataset("H1/snr", data=snrs_b)
            fb.create_dataset("H1/end_time", data=times)
            fc.create_dataset("H1/snr", data=snrs_c)
            fc.create_dataset("H1/end_time", data=times)

        res = compare_trigger_parity(f1, f2, snr_rtol=1e-4)
        self.assertTrue(res["passed"])
        self.assertLess(res["max_relative_diff"], 1e-4)

    def test_arm_definitions(self):
        self.assertIn("original_cpu", ARM_NAMES)
        self.assertIn("branch_cpu", ARM_NAMES)
        self.assertIn("branch_cpu_batched", ARM_NAMES)
        self.assertIn("jax_cpu", ARM_NAMES)
        self.assertIn("jax_cpu_batched", ARM_NAMES)
        self.assertIn("jax_cuda", ARM_NAMES)
        self.assertIn("jax_cuda_batched", ARM_NAMES)
        self.assertIn("jax_cuda_diffgw", ARM_NAMES)
        self.assertEqual(DEFAULT_ARMS, ("original_cpu", "branch_cpu", "jax_cpu", "jax_cuda"))
        self.assertEqual(
            BATCHED_ARMS,
            ("original_cpu", "branch_cpu_batched", "jax_cpu_batched", "jax_cuda_batched"),
        )


if __name__ == "__main__":
    unittest.main()
