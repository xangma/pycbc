#!/usr/bin/env python3
# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Unit tests for tools/bench_inspiral_batch_sweep.py."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py

from tools.bench_inspiral_batch_sweep import (
    generate_template_bank,
    normalize_device_and_scheme,
    parse_batch_sizes,
    run_batch_sweep,
)


class TestBenchInspiralBatchSweep(unittest.TestCase):
    def test_parse_batch_sizes(self):
        self.assertEqual(parse_batch_sizes([16, 64, 128]), [16, 64, 128])
        self.assertEqual(parse_batch_sizes(["16", "64", "128"]), [16, 64, 128])
        self.assertEqual(
            parse_batch_sizes(["16,64,128", "256"]), [16, 64, 128, 256]
        )
        self.assertEqual(
            parse_batch_sizes("16, 64, 128, 256"), [16, 64, 128, 256]
        )
        self.assertEqual(
            parse_batch_sizes("16 64 128"), [16, 64, 128]
        )

        with self.assertRaises(ValueError):
            parse_batch_sizes([0, 16])

        with self.assertRaises(ValueError):
            parse_batch_sizes("-16")

    def test_normalize_device_and_scheme(self):
        dev, scheme, cuda_idx = normalize_device_and_scheme("cuda:0")
        self.assertEqual(dev, "cuda:0")
        self.assertEqual(scheme, "torch:cuda:0")
        self.assertEqual(cuda_idx, 0)

        dev, scheme, cuda_idx = normalize_device_and_scheme("cuda")
        self.assertEqual(dev, "cuda:0")
        self.assertEqual(scheme, "torch:cuda:0")
        self.assertEqual(cuda_idx, 0)

        dev, scheme, cuda_idx = normalize_device_and_scheme("cuda:2")
        self.assertEqual(dev, "cuda:2")
        self.assertEqual(scheme, "torch:cuda:2")
        self.assertEqual(cuda_idx, 2)

        dev, scheme, cuda_idx = normalize_device_and_scheme("cpu")
        self.assertEqual(dev, "cpu")
        self.assertEqual(scheme, "torch:cpu:1")
        self.assertIsNone(cuda_idx)

        dev, scheme, cuda_idx = normalize_device_and_scheme("torch:cpu:1")
        self.assertEqual(dev, "cpu")
        self.assertEqual(scheme, "torch:cpu:1")
        self.assertIsNone(cuda_idx)

        dev, scheme, cuda_idx = normalize_device_and_scheme("torch:cuda:1")
        self.assertEqual(dev, "cuda:1")
        self.assertEqual(scheme, "torch:cuda:1")
        self.assertEqual(cuda_idx, 1)

    def test_generate_template_bank_taylorf2(self):
        with tempfile.TemporaryDirectory() as td:
            bank_path = Path(td) / "test_bank.hdf"
            generate_template_bank(bank_path, 32, "TaylorF2", f_lower=30.0)
            self.assertTrue(bank_path.is_file())

            with h5py.File(bank_path, "r") as f:
                self.assertEqual(len(f["mass1"]), 32)
                self.assertEqual(len(f["mass2"]), 32)
                self.assertEqual(len(f["spin1z"]), 32)
                self.assertEqual(len(f["spin2z"]), 32)
                self.assertEqual(len(f["f_lower"]), 32)
                self.assertEqual(f.attrs["approximant"], "TaylorF2")
                self.assertIn("parameters", f.attrs)

    def test_generate_template_bank_imrphenom(self):
        with tempfile.TemporaryDirectory() as td:
            bank_path = Path(td) / "test_bank_imr.hdf"
            generate_template_bank(bank_path, 16, "IMRPhenomD", f_lower=25.0)
            self.assertTrue(bank_path.is_file())

            with h5py.File(bank_path, "r") as f:
                self.assertEqual(len(f["mass1"]), 16)
                self.assertEqual(f["f_lower"][0], 25.0)
                self.assertEqual(f.attrs["approximant"], "IMRPhenomD")

    def test_run_batch_sweep_end_to_end_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "receipt.json"
            receipt = run_batch_sweep(
                batch_sizes=[4],
                device="cpu",
                approximant="TaylorF2",
                num_templates=4,
                output=out_json,
                duration=256,
                segment_length=256,
            )
            self.assertEqual(receipt["schema_version"], 2)

            self.assertTrue(out_json.is_file())
            with open(out_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["schema_version"], 2)
            self.assertEqual(loaded["benchmark_type"], "inspiral_batch_sweep")
            self.assertEqual(len(loaded["results"]), 1)

            res = loaded["results"][0]
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["batch_size"], 4)
            self.assertIn("wall_time_sec", res)
            self.assertIn("calc_time_sec", res)
            self.assertIn("setup_time_sec", res)
            self.assertIn("peak_vram_bytes", res)
            self.assertIn("templates_per_sec_wall", res)
            self.assertIn("template_seconds_per_wall_sec", res)
            self.assertGreater(res["wall_time_sec"], 0)
            self.assertGreaterEqual(res["calc_time_sec"], 0)
            self.assertGreaterEqual(res["setup_time_sec"], 0)

    def test_run_batch_sweep_imrphenomd_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "receipt_imrd.json"
            receipt = run_batch_sweep(
                batch_sizes=[4],
                device="cpu",
                approximant="IMRPhenomD",
                num_templates=4,
                output=out_json,
                duration=256,
                segment_length=256,
            )
            self.assertEqual(receipt["schema_version"], 2)
            self.assertEqual(receipt["results"][0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
