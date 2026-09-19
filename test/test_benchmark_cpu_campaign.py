"""Standard-library tests; no PyCBC, LALSuite, or benchmark dependencies."""

import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

RUNNER = (
    Path(__file__).resolve().parents[1] / "tools/benchmark_cpu_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_cpu_campaign", RUNNER)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def topology_row(cpu, package, core, siblings, allowed=True, die=-1):
    return {
        "cpu": cpu,
        "package_id": package,
        "core_id": core,
        "die_id": die,
        "smt_siblings": siblings,
        "allowed": allowed,
    }


class TopologyTests(unittest.TestCase):
    def test_sysfs_discovery_preserves_disallowed_siblings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "online").write_text("0-7\n")
            for cpu in range(8):
                directory = root / ("cpu%d/topology" % cpu)
                directory.mkdir(parents=True)
                (directory / "physical_package_id").write_text(
                    str((cpu % 4) // 2)
                )
                (directory / "core_id").write_text(str(cpu % 2))
                (directory / "thread_siblings_list").write_text(
                    "%d,%d" % (cpu % 4, cpu % 4 + 4)
                )
            rows = campaign.discover_topology(root, allowed={1, 2, 3, 5, 7})
        self.assertEqual(campaign.select_cpus(rows, "physical"), [1, 2, 3])
        self.assertEqual(campaign.select_cpus(rows, "2"), [1, 2])
        self.assertEqual(rows[2]["smt_siblings"], [2, 6])
        self.assertFalse(rows[6]["allowed"])
        for workers in ("0", "4", "invalid", "-1"):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                campaign.select_cpus(rows, workers)

    def test_package_and_die_distinguish_reused_core_ids(self):
        rows = [
            topology_row(0, 0, 0, [0]),
            topology_row(1, 1, 0, [1]),
            topology_row(2, 1, 0, [2], die=1),
        ]
        self.assertEqual(campaign.select_cpus(rows, "physical"), [0, 1, 2])

    def test_inconsistent_sibling_identity_rejected(self):
        rows = [topology_row(0, 0, 0, [0, 1]), topology_row(1, 0, 1, [0, 1])]
        with self.assertRaises(ValueError):
            campaign.select_cpus(rows, "physical")

    def test_cpu_list(self):
        self.assertEqual(
            campaign.cpu_list("0-2,8,10-11\n"), [0, 1, 2, 8, 10, 11]
        )
        for value in ("3-1", "1-2-3", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                campaign.cpu_list(value)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.topology = [
            topology_row(0, 0, 0, [0, 1]),
            topology_row(1, 0, 0, [0, 1], allowed=False),
            topology_row(2, 0, 1, [2]),
        ]

    def interval(self, percents):
        before = {
            "monotonic_seconds": 0,
            "cpu_ticks": {
                str(cpu): {"total": 100, "idle": 80} for cpu in range(3)
            },
        }
        after = {
            "monotonic_seconds": 5,
            "cpu_ticks": {
                str(cpu): {"total": 200, "idle": 180 - percent}
                for cpu, percent in enumerate(percents)
            },
        }
        return campaign.utilization(before, after)

    def test_host_uses_all_cpus_and_sibling_can_reject(self):
        interval = self.interval([0, 10, 0])
        self.assertAlmostEqual(interval["host_percent"], 10 / 3)
        assessment = campaign.idle_assessment([interval], self.topology, [0])
        self.assertFalse(assessment["passed"])
        self.assertEqual(
            assessment["monitored_cpus_including_siblings"], [0, 1]
        )
        self.assertEqual(
            assessment["reasons"],
            ["Selected CPU or SMT sibling utilization exceeded 5%"],
        )

    def test_unselected_host_work_rejects_and_threshold_is_strict(self):
        busy = campaign.idle_assessment(
            [self.interval([0, 0, 60])], self.topology, [0]
        )
        self.assertEqual(
            busy["reasons"], ["Whole-host utilization exceeded 5%"]
        )
        self.assertTrue(
            campaign.idle_assessment(
                [self.interval([5, 5, 5])], self.topology, [0]
            )["passed"]
        )

    def test_short_or_incomplete_samples_fail_closed(self):
        interval = self.interval([0, 0, 0])
        interval["interval_seconds"] = 4.99
        self.assertFalse(
            campaign.idle_assessment([interval], self.topology, [0])["passed"]
        )
        interval["interval_seconds"] = 5
        del interval["per_cpu_percent"]["2"]
        self.assertFalse(
            campaign.idle_assessment([interval], self.topology, [0])["passed"]
        )
        self.assertFalse(
            campaign.idle_assessment([], self.topology, [0])["passed"]
        )

    def test_counter_reset_fails_closed(self):
        before = {
            "monotonic_seconds": 0,
            "cpu_ticks": {"0": {"total": 50, "idle": 30}},
        }
        after = {
            "monotonic_seconds": 5,
            "cpu_ticks": {"0": {"total": 40, "idle": 20}},
        }
        self.assertFalse(campaign.utilization(before, after)["complete"])

    def test_labels_never_infer_exclusive_reservation(self):
        self.assertEqual(
            campaign.host_label(False, "", True), "observational_idle"
        )
        self.assertEqual(
            campaign.host_label(False, "job 123", True),
            "user_asserted_reservation_with_observed_idle",
        )
        self.assertEqual(
            campaign.host_label(False, "job 123", False),
            "rejected_host_checks",
        )
        for passed in (True, False):
            self.assertEqual(
                campaign.host_label(True, "job 123", passed), "shared_host"
            )

    def test_proc_capture_excludes_double_counted_guest_ticks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stat").write_text(
                "cpu 10 0 10 80 0 0 0 0 9 0\n" "cpu0 10 0 10 80 0 0 0 0 9 0\n"
            )
            (root / "meminfo").write_text(
                "MemTotal: 1000 kB\nMemAvailable: 700 kB\n"
            )
            (root / "loadavg").write_text("0.01 0.02 0.03 1/12 123\n")
            directory = root / "123"
            directory.mkdir()
            fields = ["0"] * 37
            fields[0], fields[1], fields[2] = "S", "1", "123"
            fields[11], fields[12], fields[17], fields[36] = (
                "10",
                "5",
                "2",
                "0",
            )
            (directory / "stat").write_text(
                "123 (name (with) spaces) " + " ".join(fields)
            )
            captured = campaign.snapshot(root)
        self.assertEqual(
            captured["cpu_ticks"]["0"], {"total": 100, "idle": 80}
        )
        self.assertEqual(captured["memory"]["MemAvailable"], "700 kB")
        self.assertEqual(
            captured["processes"][0]["name"], "name (with) spaces"
        )
        self.assertEqual(captured["processes"][0]["threads"], 2)
        self.assertEqual(captured["processes"][0]["cpu_ticks"], 15)

    def test_samples_persist_before_and_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            first = {
                "monotonic_seconds": 0,
                "cpu_ticks": {"0": {"total": 100, "idle": 50}},
            }
            last = {
                "monotonic_seconds": 1,
                "cpu_ticks": {"0": {"total": 200, "idle": 100}},
            }
            with mock.patch.object(
                campaign, "snapshot", side_effect=[first, last]
            ):
                sampler = campaign.Sampler(path)
                sampler.take("before")
                sampler.take("after")
            saved = [
                json.loads(line) for line in path.read_text().splitlines()
            ]
        self.assertEqual([row["phase"] for row in saved], ["before", "after"])
        self.assertEqual(saved[1]["utilization"]["host_percent"], 50)


class MetricsAndCommandTests(unittest.TestCase):
    def test_aggregate_divides_by_makespan_and_allocated_physical_cores(self):
        result = campaign.throughput_metrics(
            [200, 200], makespan=4, physical_cores=2
        )
        self.assertEqual(
            result["template_seconds_per_wall_second_per_physical_core"], 50
        )
        self.assertEqual(result["allocated_physical_cores"], 2)
        self.assertEqual(result["total_template_seconds"], 400)
        # Summing each worker's throughput (100 + 50) would be wrong.
        self.assertNotEqual(
            result["template_seconds_per_wall_second_per_physical_core"],
            200 / 2 + 200 / 4,
        )
        for seconds, cores in ((0, 2), (1, 0)):
            with self.assertRaises(ValueError):
                campaign.throughput_metrics([200], seconds, cores)

    def test_source_metadata_and_executable_symlink_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "venv-python"
            executable.symlink_to(sys.executable)
            path = root / "argv.json"
            metadata = {
                "git_commit": "abc123",
                "source_sha256": "declared hash",
            }
            spec = {
                "argv": [str(executable), "-c", "print('{}')"],
                "metadata": metadata,
            }
            path.write_text(json.dumps(spec))
            argv, provenance = campaign.command_spec(path)
        self.assertEqual(argv, spec["argv"])
        self.assertEqual(
            provenance["supplied_command_spec"]["metadata"], metadata
        )
        self.assertEqual(len(provenance["command_file_sha256"]), 64)

    def test_invalid_commands_and_numeric_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "argv.json"
            for spec in (
                [],
                "echo hi",
                {"argv": "echo hi"},
                [sys.executable, 1],
                ["a\0b"],
            ):
                path.write_text(json.dumps(spec))
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    campaign.command_spec(path)
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(
                campaign.argparse.ArgumentTypeError
            ):
                campaign.positive(value)


class CliTests(unittest.TestCase):
    def exercise(self, percent, extra=()):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = root / "command.json"
            command.write_text(json.dumps([sys.executable, "-c", "pass"]))
            output = root / "campaign"
            elapsed = [0.0]
            clock = mock.Mock()
            clock.monotonic.side_effect = lambda: elapsed[0]
            clock.sleep.side_effect = lambda seconds: elapsed.__setitem__(
                0, elapsed[0] + seconds
            )
            sampler = mock.Mock()
            sampler.take.return_value = {
                "utilization": {
                    "interval_seconds": 1.0,
                    "complete": True,
                    "host_percent": percent,
                    "per_cpu_percent": {"0": percent},
                }
            }
            rows = [topology_row(0, 0, 0, [0])]
            with mock.patch.object(
                campaign.sys, "platform", "linux"
            ), mock.patch.object(campaign, "time", clock), mock.patch.object(
                campaign, "discover_topology", return_value=rows
            ), mock.patch.object(
                campaign, "Sampler", return_value=sampler
            ), mock.patch.object(
                campaign, "run_repeat", return_value={"status": "completed"}
            ) as run, mock.patch(
                "sys.stdout", new_callable=io.StringIO
            ):
                status = campaign.main(
                    [
                        "--command-json",
                        str(command),
                        "--output",
                        str(output),
                        "--templates",
                        "100",
                        "--valid-seconds",
                        "10",
                        *extra,
                    ]
                )
            report = json.loads((output / "receipt.json").read_text())
            self.assertGreaterEqual(elapsed[0], 5)
            return status, report, run

    def test_strict_default_rejects_busy_host_before_launch(self):
        status, report, run = self.exercise(50)
        self.assertEqual(status, 1)
        self.assertEqual(report["host_label"], "rejected_host_checks")
        self.assertFalse(report["preflights"][0]["passed"])
        self.assertEqual(len(report["preflights"][0]["intervals"]), 5)
        run.assert_not_called()

    def test_shared_host_explicitly_bypasses_idle_check_and_keeps_label(self):
        status, report, run = self.exercise(
            50, ["--shared-host", "--reservation-note", "operator note"]
        )
        self.assertEqual(status, 0)
        self.assertEqual(report["host_label"], "shared_host")
        self.assertFalse(report["preflights"][0]["passed"])
        run.assert_called_once()

    def test_idle_is_observational_and_reservation_does_not_bypass_busy(self):
        status, report, run = self.exercise(0)
        self.assertEqual(status, 0)
        self.assertEqual(report["host_label"], "observational_idle")
        run.assert_called_once()
        status, report, run = self.exercise(
            50, ["--reservation-note", "job 123"]
        )
        self.assertEqual(status, 1)
        self.assertEqual(report["host_label"], "rejected_host_checks")
        run.assert_not_called()


# macOS can exercise the real process/barrier/group orchestration. Only the
# Linux affinity syscall is substituted in the child; Linux uses the real one.
PORTABLE_LAUNCHER = """
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location('campaign', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cpu, ready, gate = map(int, sys.argv[2:])
if sys.platform != 'linux':
    os.sched_setaffinity = lambda pid, cpus: None
    os.sched_getaffinity = lambda pid: {cpu}
module.worker_main(cpu, ready, gate)
"""


@unittest.skipUnless(os.name == "posix", "Requires POSIX process groups")
class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "output dir; $not_expanded"
        self.root.mkdir()
        self.cpus = (
            sorted(os.sched_getaffinity(0))[:2]
            if sys.platform == "linux"
            else [0, 1]
        )
        patcher = mock.patch.object(
            campaign,
            "worker_argv",
            side_effect=lambda cpu, ready, gate: [
                sys.executable,
                "-B",
                "-c",
                PORTABLE_LAUNCHER,
                str(RUNNER),
                str(cpu),
                str(ready),
                str(gate),
            ],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_command(self, code, timeout=5, cpus=None):
        self.phases = []
        return campaign.run_repeat(
            [sys.executable, "-c", code, "{output_dir}"],
            self.root / "repeat",
            self.cpus if cpus is None else cpus,
            100,
            timeout,
            self.phases.append,
        )

    def assert_stopped(self, pid):
        try:
            if sys.platform == "linux":
                stat = Path("/proc/%d/stat" % pid)
                for _ in range(100):
                    if (
                        not stat.exists()
                        or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
                    ):
                        return
                    time.sleep(0.01)
            else:
                for _ in range(100):
                    result = subprocess.run(
                        ["ps", "-p", str(pid), "-o", "stat="],
                        capture_output=True,
                        text=True,
                    )
                    if (
                        not result.stdout.strip()
                        or result.stdout.strip().startswith("Z")
                    ):
                        return
                    time.sleep(0.01)
            self.fail("Own child %d survived cleanup" % pid)
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_concurrent_barrier_unique_outputs_environment_and_full_wall_time(
        self,
    ):
        code = """
import json, os, pathlib, sys, time
directory = pathlib.Path(sys.argv[1])
assert directory == pathlib.Path.cwd()
time.sleep(0.15)
data = {'start': time.monotonic(), 'omp': os.environ['OMP_NUM_THREADS'],
        'mkl_dynamic': os.environ['MKL_DYNAMIC'], 'literal': '{}'}
if hasattr(os, 'sched_getaffinity'):
    data['affinity'] = sorted(os.sched_getaffinity(0))
(directory / 'result.json').write_text(json.dumps(data))
print('complete', flush=True)
"""
        with mock.patch.dict(
            os.environ, {"OMP_NUM_THREADS": "64", "MKL_DYNAMIC": "TRUE"}
        ):
            result = self.run_command(code)
        self.assertEqual(result["status"], "completed", result)
        self.assertIn("during", self.phases)
        self.assertEqual(
            len({row["cwd"] for row in result["workers"]}), len(self.cpus)
        )
        for row in result["workers"]:
            data = json.loads((Path(row["cwd"]) / "result.json").read_text())
            self.assertEqual(
                (data["omp"], data["mkl_dynamic"], data["literal"]),
                ("1", "FALSE", "{}"),
            )
            self.assertGreaterEqual(row["wall_seconds"], 0.15)
            self.assertGreater(
                data["start"], result["barrier_release_monotonic_seconds"]
            )
            self.assertEqual(Path(row["stdout"]).read_text(), "complete\n")
            if sys.platform == "linux":
                self.assertEqual(data["affinity"], [row["cpu"]])
        metrics = result["metrics"]
        self.assertAlmostEqual(
            metrics["template_seconds_per_wall_second_per_physical_core"],
            100 / metrics["makespan_seconds"],
        )
        self.assertGreaterEqual(metrics["makespan_seconds"], 0.15)
        self.assertLess(metrics["makespan_seconds"], 3)

    def descendant_code(self, ending):
        return """
import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
    'time.sleep(60)'])
pathlib.Path('child.pid').write_text(str(child.pid))
time.sleep(0.1)
""" + ending

    def test_timeout_kills_entire_groups_and_saves_failed_receipt(self):
        result = self.run_command(
            self.descendant_code("time.sleep(60)"), timeout=0.8
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("TimeoutError", result["error"])
        self.assertIsNone(result["metrics"])
        saved = json.loads((self.root / "repeat/receipt.json").read_text())
        self.assertEqual(saved["status"], "failed")
        for row in result["workers"]:
            self.assert_stopped(row["pid"])
            self.assert_stopped(
                int((Path(row["cwd"]) / "child.pid").read_text())
            )

    def test_nonzero_exit_cleans_descendants_and_aborts_other_workers(self):
        code = self.descendant_code(
            "if pathlib.Path.cwd().name.endswith('000'): sys.exit(7)\n"
            "time.sleep(60)"
        )
        before = time.monotonic()
        result = self.run_command(code)
        self.assertLess(time.monotonic() - before, 4)
        self.assertEqual(result["status"], "failed")
        self.assertIn(7, [row["returncode"] for row in result["workers"]])
        for row in result["workers"]:
            self.assert_stopped(row["pid"])
            self.assert_stopped(
                int((Path(row["cwd"]) / "child.pid").read_text())
            )

    def test_successful_parent_leaving_descendants_is_failed_and_cleaned(self):
        result = self.run_command(
            self.descendant_code("sys.exit(0)"), cpus=self.cpus[:1]
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("descendants", result["error"])
        self.assert_stopped(
            int((Path(result["workers"][0]["cwd"]) / "child.pid").read_text())
        )

    def test_failed_executable_receipt_and_barrier_failure_cleanup(self):
        with mock.patch.object(
            campaign,
            "worker_argv",
            return_value=[sys.executable, "-c", "import sys; sys.exit(9)"],
        ):
            result = self.run_command("pass")
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("barrier_release_monotonic_seconds", result)
        for row in result["workers"]:
            self.assert_stopped(row["pid"])

    def test_timeout_before_barrier_cleans_waiting_workers(self):
        with mock.patch.object(
            campaign,
            "worker_argv",
            return_value=[sys.executable, "-c", "import time; time.sleep(60)"],
        ):
            result = self.run_command("pass", timeout=0.2)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Timed out preparing", result["error"])
        for row in result["workers"]:
            self.assert_stopped(row["pid"])

    def test_sampling_failure_cleans_running_workers(self):
        def failure(phase):
            raise OSError("simulated sampling failure")

        result = campaign.run_repeat(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            self.root / "repeat",
            self.cpus,
            100,
            5,
            failure,
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("simulated sampling failure", result["error"])
        for row in result["workers"]:
            self.assert_stopped(row["pid"])


if __name__ == "__main__":
    unittest.main()
