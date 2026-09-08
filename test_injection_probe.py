"""Synthetic checks for the probe; these do not validate the remote capture."""

import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "injection_probe", Path(__file__).with_name("injection-probe.py")
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
np = probe.np

from pycbc import scheme
from pycbc.fft.backend_support import set_backend
from pycbc.filter import matchedfilter as mf
from pycbc.vetoes import chisq

REPO = Path(__file__).resolve().parents[2]


def fixture(root, size=8192):
    """Create an internally captured synthetic CPU fixture, with no strain file."""
    root.mkdir()
    rng = np.random.default_rng(20260908)
    df = 1024 / size
    lo, hi = int(30 / df), size // 2
    psd = (1.0 + np.arange(hi + 1) / hi).astype(np.float32)
    meta = dict(
        status="complete",
        arrays={},
        templates={},
        points=[],
        strain=dict(delta_t=1 / 1024, epoch="1187007048.0"),
        segment_spectra=[],
        sources={},
        segments=[
            dict(
                start=i * size // 2,
                stop=i * size // 2 + size,
                analyze_start=size // 4,
                analyze_stop=size - 100,
            )
            for i in range(5)
        ],
    )

    def save(name, array):
        array = np.asarray(array)
        np.save(root / name, array, allow_pickle=False)
        meta["arrays"][name] = dict(
            sha256=probe.sha(root / name),
            data_sha256=probe.array_sha(array),
            shape=list(array.shape),
            dtype=str(array.dtype),
        )
        return name

    save("psd-proposed.npy", psd)
    save("psd-output-0.npy", psd)
    for i in range(5):
        noise = (0.3 * (rng.standard_normal(hi + 1) + 1j * rng.standard_normal(hi + 1))).astype(
            np.complex64
        )
        noise[[0, -1]] = 0
        meta["segment_spectra"].append(save(f"spectrum-{i}.npy", noise))
    legacy, _ = probe.original_functions(REPO)
    for ordinal, (index, segment) in enumerate(probe.SELECTED):
        frequency = np.arange(hi + 1)
        h = (
            (1 + frequency / (40 + ordinal * 5)) ** (-0.3)
            * np.exp(1j * (frequency * 0.11 + ordinal))
        ).astype(np.complex64)
        h[:lo] = 0
        h[-1] = 0
        info = dict(
            kmin=lo,
            kmax=hi,
            delta_f=df,
            num_bins=16,
            template_hash=index + 100,
            file=save(f"template-{index}.npy", h),
        )
        native = probe.NativeFilter(h, psd, info)
        bins = probe.bins_for(native, info, legacy)["proposed"]
        prefix = mf.sigmasq_series(native.template, native.psd, lo * df, hi * df).numpy()
        info.update(
            captured_current_bins=bins.tolist(),
            current_prefix_file=save(f"prefix-{index}.npy", prefix),
        )
        meta["templates"][str(index)] = info
        noise = np.load(root / meta["segment_spectra"][segment])
        corr, snr = native.apply(noise)
        indices = np.asarray([size // 2, size - 600 - ordinal * 11], dtype=np.uint32)
        q, _ = probe.calibration(h, psd, lo, hi, df)
        norm = float(4 * df / np.sqrt(q))
        chi = chisq.power_chisq_at_points_from_precomputed(
            native.corr, snr[indices], norm, bins, indices
        )
        meta["points"].append(
            dict(
                index=index,
                segment=segment,
                norm=norm,
                bins=bins.tolist(),
                correlation_file=save(f"corr-{index}.npy", corr),
                snr_file=save(f"snr-{index}.npy", snr[indices]),
                indices_file=save(f"indices-{index}.npy", indices),
                chisq_file=save(f"chi-{index}.npy", chi),
            )
        )
    fn = chisq.power_chisq_bins
    meta["sources"][fn.__module__ + "." + fn.__qualname__] = dict(
        source_sha256=probe.hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    )
    (root / "capture.json").write_text(json.dumps(meta))
    return meta, legacy


class InjectionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = scheme.CPUScheme(1)
        cls.context.__enter__()
        set_backend(["numpy"])
        cls.folder = tempfile.TemporaryDirectory(prefix="pycbc-injection-fixture-")
        cls.root = Path(cls.folder.name) / "capture"
        cls.meta, cls.legacy = fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()
        cls.context.__exit__(None, None, None)

    def test_all_48_cases_null_gates_calibration_timing_and_statistic(self):
        capture = probe.Capture(self.root, self.root / "capture.json")
        result = {}
        with contextlib.redirect_stdout(io.StringIO()):
            probe.probe(capture, self.legacy, result, lambda: None)
        self.assertEqual(len(result["cases"]), 48)
        self.assertEqual(len(result["null_checks"]), 6)
        self.assertTrue(result["null_gate_passed"])
        self.assertTrue(capture.unchanged())
        for null in result["null_checks"]:
            self.assertTrue(all(null["checks"].values()))
            self.assertEqual(null["selected_index"], max(null["captured_indices"]))
        for row in result["cases"]:
            for point in row["points"]:
                self.assertLess(abs(point["reference_fft_minus_direct_snr"]), 1e-10)
                self.assertLess(abs(point["variants"]["proposed"]["evaluator_error"]), 2e-4)
            if row["mode"] == "signal_only":
                self.assertEqual(row["peak_index"], row["injected_index"])
                self.assertEqual(row["reference_peak_index"], row["injected_index"])
                self.assertLess(row["target_calibration_relative_error"], 1e-7)
                target = next(p for p in row["points"] if p["index"] == row["injected_index"])
                for label in ("original", "proposed"):
                    self.assertAlmostEqual(
                        target["variants"][label]["reference"]["chisq"],
                        row["analytic_target_chisq"][label],
                        delta=2e-6,
                    )

    def test_wrong_null_aborts_before_any_injection(self):
        capture = probe.Capture(self.root, self.root / "capture.json")
        # Preserve all files and hashes; corrupt only an expected bin in memory.
        capture.metadata["points"][-1]["bins"][1] += 1
        result = {}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            ValueError, "Capture null failed"
        ):
            probe.probe(capture, self.legacy, result, lambda: None)
        self.assertEqual(result["cases"], [])
        self.assertEqual(len(result["null_checks"]), 5)
        self.assertFalse(result.get("null_gate_passed", False))

    def test_hash_failure_rejected(self):
        capture = probe.Capture(self.root, self.root / "capture.json")
        capture.metadata["arrays"]["psd-proposed.npy"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "File hash mismatch"):
            capture.load("psd-proposed.npy")

    def test_independent_flat_band_solution_and_destructive_noise(self):
        size, lo, hi, index, df = 8192, 240, 4096, 7801, 0.125
        h = np.zeros(size // 2 + 1, dtype=np.complex64)
        h[lo:hi] = 2 + 1j
        psd = np.full(len(h), 4, dtype=np.float32)
        q, _ = probe.calibration(h, psd, lo, hi, df)
        self.assertEqual(q, 4 * df * (hi - lo) * 5 / 4)
        spectrum = probe.injected_input(h, None, 12.0, q, index, lo, hi)
        c = probe.reference_correlation(h, spectrum, psd, lo, hi)
        z = probe.fft.ifft(c, norm="forward")
        self.assertEqual(int(np.argmax(abs(z))), index)
        self.assertAlmostEqual(abs(z[index]) * float(4 * df / np.sqrt(q)), 12.0, delta=1e-6)
        cancelled = probe.injected_input(h, -spectrum, 12.0, q, index, lo, hi)
        residual = probe.reference_correlation(h, cancelled, psd, lo, hi)
        zr = probe.fft.ifft(residual, norm="forward")
        self.assertLess(abs(zr[index]) * float(4 * df / np.sqrt(q)), 1e-6)

    def test_time_phase_integer_range_reduction(self):
        size, index = 2097152, 1654835
        phase = probe.phase(15360, 15380, index, size, probe.LD)
        for offset, value in enumerate(phase):
            k = 15360 + offset
            theta = probe.LD(2) * np.arccos(probe.LD(-1)) * probe.LD((k * index) % size) / size
            self.assertLess(abs(value - (np.cos(theta) + 1j * np.sin(theta))), 1e-14)

    def test_real_capture_metadata_contract_without_bulk_arrays(self):
        meta = json.loads(Path(__file__).with_name("capture-metadata.json").read_text())
        self.assertEqual([(p["index"], p["segment"]) for p in meta["points"]], probe.SELECTED)
        self.assertEqual(meta["strain"]["delta_t"], 1 / 4096)
        for point in meta["points"]:
            info = meta["templates"][str(point["index"])]
            self.assertEqual(meta["arrays"][info["file"]]["dtype"], "complex64")
            self.assertEqual(meta["arrays"][point["correlation_file"]]["shape"], [2097152])
            self.assertEqual(info["kmin"] * info["delta_f"], 30.0)
            self.assertEqual(info["captured_current_bins"], point["bins"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
