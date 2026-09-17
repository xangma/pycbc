"""Unit tests for pycbc.hardware cache hierarchy detection utilities."""

import os
import unittest
from pycbc.hardware import (
    _parse_cache_size_string,
    get_cpu_l3_cache_size,
    get_gpu_l2_cache_size,
    get_optimal_batch_maxelements,
)


class TestHardwareCacheDetection(unittest.TestCase):
    def test_parse_cache_size_string(self):
        self.assertEqual(_parse_cache_size_string("32M"), 32 * 1024 * 1024)
        self.assertEqual(_parse_cache_size_string("16384K"), 16384 * 1024)
        self.assertEqual(_parse_cache_size_string("1G"), 1024 * 1024 * 1024)
        self.assertEqual(_parse_cache_size_string("72MB"), 72 * 1024 * 1024)
        self.assertEqual(_parse_cache_size_string(""), 0)
        self.assertEqual(_parse_cache_size_string(None), 0)

    def test_get_cpu_l3_cache_size(self):
        per_ccx = get_cpu_l3_cache_size(per_ccx=True)
        total = get_cpu_l3_cache_size(per_ccx=False)
        self.assertGreater(per_ccx, 0)
        self.assertGreater(total, 0)
        self.assertGreaterEqual(total, per_ccx)

    def test_get_gpu_l2_cache_size(self):
        l2_size = get_gpu_l2_cache_size(device_id=0)
        self.assertGreaterEqual(l2_size, 1024 * 1024)

    def test_get_optimal_batch_maxelements_defaults(self):
        cpu_max = get_optimal_batch_maxelements(is_cuda=False)
        cuda_max = get_optimal_batch_maxelements(is_cuda=True)
        self.assertGreaterEqual(cpu_max, 2**21)
        self.assertGreaterEqual(cuda_max, 2**19)
        self.assertLessEqual(cuda_max, 2**25)

    def test_get_optimal_batch_maxelements_env_override(self):
        old_env = os.environ.get("PYCBC_BATCH_MAXELEMENTS")
        try:
            os.environ["PYCBC_BATCH_MAXELEMENTS"] = "1234567"
            self.assertEqual(get_optimal_batch_maxelements(is_cuda=True), 1234567)
            self.assertEqual(get_optimal_batch_maxelements(is_cuda=False), 1234567)
        finally:
            if old_env is None:
                os.environ.pop("PYCBC_BATCH_MAXELEMENTS", None)
            else:
                os.environ["PYCBC_BATCH_MAXELEMENTS"] = old_env


if __name__ == "__main__":
    unittest.main()
