"""Focused tests for the JAX device-memory profiling wrapper."""

import importlib.util
import logging
import sys
from pathlib import Path
import types
import unittest
import weakref
from unittest import mock


_SPEC = importlib.util.spec_from_file_location(
    "profile_jax_memory",
    Path(__file__).resolve().parents[1] / "tools/profile_jax_memory.py",
)
profiler = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profiler)


class _Array:
    shape = (2, 3)
    dtype = "float32"
    nbytes = 24

    def __init__(self, pointer):
        self.pointer = pointer
        self.ready = 0

    def unsafe_buffer_pointer(self):
        return self.pointer

    def block_until_ready(self):
        self.ready += 1

    @property
    def device(self):
        return "Fake GPU"


class _DeviceWithoutStats:
    def __str__(self):
        return "fake-device"


class MemoryInventoryTest(unittest.TestCase):
    def test_pointer_aliases_are_grouped_without_content_claim(self):
        arrays = [_Array(100), _Array(100), _Array(None)]
        inventory = profiler.inventory_live_arrays(arrays)

        self.assertEqual(inventory["array_count"], 3)
        self.assertEqual(inventory["same_pointer_alias_groups"], [{
            "device": "Fake GPU",
            "unsafe_buffer_pointer": 100,
            "array_indices": [0, 1],
            "nbytes": [24, 24],
        }])
        self.assertIsNone(inventory["arrays"][2]["unsafe_buffer_pointer"])
        self.assertNotIn("content", inventory["same_pointer_alias_groups"][0])

    def test_unavailable_allocator_stats_remain_null(self):
        jax = types.SimpleNamespace(devices=lambda: [_DeviceWithoutStats()])
        stats = profiler.device_memory_stats(jax)

        self.assertEqual(stats[0]["device"], "fake-device")
        self.assertIsNone(stats[0]["memory_stats"])


class SnapshotLifetimeTest(unittest.TestCase):
    def test_snapshot_serializes_metadata_and_drops_array_reference(self):
        output_dir = self.create_temp_dir()
        array_ref = []

        def live_arrays():
            array = _Array(200)
            array_ref.append(weakref.ref(array))
            return [array]

        jax = types.ModuleType("jax")
        jax.__path__ = []
        jax.live_arrays = live_arrays
        jax.devices = lambda: [_DeviceWithoutStats()]
        jax_profiler = types.ModuleType("jax.profiler")

        def save_device_memory_profile(path):
            Path(path).write_bytes(b"fake pprof")

        jax_profiler.save_device_memory_profile = save_device_memory_profile
        jax.profiler = jax_profiler
        with mock.patch.dict(
            sys.modules, {"jax": jax, "jax.profiler": jax_profiler}
        ):
            instance = profiler.JAXMemoryProfiler(output_dir)
            instance.capture("batch", "test trigger")

        self.assertEqual(
            instance.snapshots[0]["live_arrays"]["array_count"], 1
        )
        self.assertIsNone(
            instance.snapshots[0]["device_memory_stats"][0]["memory_stats"]
        )
        self.assertTrue((output_dir / "batch.pprof").exists())
        self.assertIsNone(array_ref[0]())
        self.assertNotIn("arrays", vars(instance))

    def create_temp_dir(self):
        self.addCleanup(self._tempdir.cleanup)
        return Path(self._tempdir.name)

    def setUp(self):
        import tempfile

        self._tempdir = tempfile.TemporaryDirectory()


class LoggingHookTest(unittest.TestCase):
    def test_selects_first_middle_final_distinct_batches_and_cleanup(self):
        captures = []

        class FakeProfiler:
            def capture(self, label, trigger):
                captures.append((label, trigger))

        hook = profiler.LoggingCaptureHook(FakeProfiler())
        for first in range(1, 13):
            hook(
                "Filtering template batch %d-%d/48 segment 1/3",
                first,
                first + 3,
            )
            hook(
                "Filtering template batch %d-%d/48 segment 2/3",
                first,
                first + 3,
            )
        hook("We currently have %d triggers", 2)
        hook("We currently have %d triggers", 3)

        self.assertEqual(
            [label for label, _ in captures],
            [
                "production_batch_1_segment_1",
                "production_batch_6_segment_1",
                "production_batch_12_segment_1",
                "postcleanup",
            ],
        )

    def test_named_logger_info_is_observed(self):
        captures = []

        class FakeProfiler:
            def capture(self, label, trigger):
                captures.append((label, trigger))

        hook = profiler.LoggingCaptureHook(FakeProfiler())
        old_info = logging.Logger.info

        def logger_info(logger, message, *args, **kwargs):
            hook.logger_info(logger, message, *args, **kwargs)

        logging.Logger.info = logger_info
        try:
            logging.getLogger("events").info(
                "We currently have %d triggers", 4
            )
        finally:
            logging.Logger.info = old_info

        self.assertEqual([label for label, _ in captures], ["postcleanup"])


if __name__ == "__main__":
    unittest.main()
