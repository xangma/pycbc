"""Regression tests for observed timeline phase boundaries."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "profile_jax_gpu_timeline",
    Path(__file__).resolve().parents[1] / "tools/profile_jax_gpu_timeline.py",
)
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)


class TimelinePhasesTest(unittest.TestCase):
    def test_actual_batches_exclude_warmup_and_support_arbitrary_sizes(self):
        logs = [
            (1, "Reading Frames"),
            (2, "Highpass Filtering"),
            (3, "Read in template bank"),
            (3.1, "Warming up JAX JIT compilation during setup"),
            (3.2, "Decompressing template batch 1-16"),
            (4, "Decompressing template batch 1-16"),
            (4.2, "Filtering template batch 1-16/20 segment 1/5"),
            (4.3, "Filtering template batch 1-16/20 segment 2/5"),
            (5, "Decompressing template batch 17-20"),
            (5.2, "Filtering template batch 17-20/20 segment 1/5"),
            (6, "We currently have 10 triggers"),
            (6.1, "Writing out triggers"),
            (6.2, "Finished"),
        ]
        phases = profiler._identify_pipeline_phases(logs, 7)
        batches = [p for p in phases if p["name"].startswith("filter_batch_")]
        self.assertEqual([p["start_sec"] for p in batches], [4, 5])
        self.assertIn("17–20", batches[1]["label"])
        self.assertEqual(phases[-1]["name"], "teardown")
        self.assertEqual(phases[-1]["start_sec"], 6.2)
        self.assertEqual(phases[-1]["end_sec"], 7)
        self.assertTrue(all("gpu_status" not in p for p in phases))
        for previous, following in zip(phases, phases[1:]):
            self.assertEqual(previous["end_sec"], following["start_sec"])

    def test_absent_milestones_do_not_create_fictional_phases(self):
        phases = profiler._identify_pipeline_phases([], 0.4)
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0]["name"], "execution")
        self.assertEqual(phases[0]["duration_sec"], 0.4)
        self.assertEqual(profiler._identify_pipeline_phases([], 0), [])

    def test_decompression_only_warmup_requires_observed_restart(self):
        logs = [
            (1, "Warming up JAX JIT compilation during setup"),
            (1.1, "Decompressing template batch 1-32"),
        ]
        phases = profiler._identify_pipeline_phases(logs, 2)
        self.assertFalse(
            any(p["name"].startswith("filter_batch_") for p in phases)
        )
        logs += [
            (2, "Decompressing template batch 1-32"),
            (3, "Decompressing template batch 33-64"),
        ]
        phases = profiler._identify_pipeline_phases(logs, 4)
        batches = [p for p in phases if p["name"].startswith("filter_batch_")]
        self.assertEqual([p["start_sec"] for p in batches], [2, 3])

    def test_filtering_without_active_decompression_uses_filtering_log(self):
        logs = [
            (1, "Warming up JAX JIT compilation during setup"),
            (1.1, "Decompressing template batch 1-32"),
            (2, "Filtering template batch 1-32/32 segment 1/5"),
        ]
        phases = profiler._identify_pipeline_phases(logs, 3)
        self.assertEqual(phases[-1]["start_sec"], 2)
        self.assertEqual(phases[-1]["name"], "filter_batch_1")

    def test_no_warmup_and_out_of_range_logs(self):
        logs = [
            (-1, "Reading Frames"),
            (0.1, "Decompressing template batch 1-8"),
            (0.2, "Decompressing template batch 9-10"),
            (3, "Finished"),
        ]
        phases = profiler._identify_pipeline_phases(logs, 1)
        self.assertEqual([p["start_sec"] for p in phases], [0, 0.1, 0.2])
        self.assertEqual(phases[-1]["end_sec"], 1)


if __name__ == "__main__":
    unittest.main()
