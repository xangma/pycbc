#!/usr/bin/env python3
"""Validate and plot a completed, frozen R4 batch campaign offline.

Usage: python tools/plot_torch_batch_sweep.py --campaign DIR --output DIR
Add --verify-only to revalidate evidence and an existing plot manifest.
Requires stage-manifest.json, aggregate JSON, runs/*/{result,acquisition}.json,
stages/*.json and the six frozen helpers. No PyCBC, Torch or .npy replay.
Outputs must be outside the archive; existing outputs are never overwritten.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import types


HELPERS = ("batch-campaign.py", "batch-worker.py", "campaign_controls.py",
           "NUMERICAL-POLICY.md", "policy-decision.json", "native-provenance.json")
LABELS = ("Standard CPU / MKL", "Torch CPU", "Torch CUDA + one GPU")
COLORS = ("#596579", "#2274A5", "#C66B18")
IMAGES = tuple(f"batch-{plot}.{ext}" for plot in ("throughput", "accuracy")
               for ext in ("png", "svg"))
SCOPE = {
    "source_revision": "9578a710479b924e882857c4dffab6ed372a634b",
    "measurement": ("Warm public LiveBatchMatchedFilter.process_data API on shared host len; "
                    "timing uninstrumented."),
    "resources": ("One numerical thread, affinity CPU 8; CUDA additionally uses one GPU (cuda:0). "
                  "No exclusive reservation."),
    "throughput": ("1024 templates x 3 blocks = 3072 template-block evaluations / second of iteration time; "
                   "median of 3 fresh-worker medians, observed min-max across workers (not a confidence interval)."),
    "accuracy": ("Maximum absolute normalized complex SNR error over all rows, samples and blocks of both "
                 "full qualification seeds 7102/7103; independent complex128 oracle and MKL compatibility; "
                 "threshold 0.001. Symlog axis, linear near zero below 1e-8; exact zeros retained."),
    "validation": ("Frozen controller JSON evidence gates and summary recomputation; does not replay .npy "
                   "arrays or re-execute science, source or native binaries."),
    "excludes": ["frame I/O", "PSD estimation", "waveform generation", "bank loading",
                 "CLI startup/teardown", "full-machine capacity"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(data, parse_constant=invalid, object_pairs_hook=unique)


class Archive:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def path(self, name):
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), f"Input escapes archive: {name}")
        return path

    def read(self, name):
        data = self.path(name).read_bytes()
        self.hashes[name] = hashlib.sha256(data).hexdigest()
        return read_json(data)

    def unchanged(self):
        for name, expected in self.hashes.items():
            require(digest(self.path(name)) == expected,
                    f"Input changed: {name}")


class OfflineReference:
    """Keep remote identity for controller comparisons, read local evidence."""

    def __init__(self, declared, local):
        self.declared = declared
        self.local = local

    def __str__(self):
        return self.declared

    def __truediv__(self, name):
        require(name == "result.json", "Unexpected reference read")
        return self.local / name


@contextmanager
def frozen_controller(archive):
    # The pre-acquisition stage manifest is the initial trust anchor; its own
    # hash is then frozen in the published plot manifest. Check before executing
    # any archived Python. Compiling the controller/control sources also ignores
    # their __pycache__ files beside the archive.
    staged = archive.read("stage-manifest.json")
    frozen = {name: staged[name] for name in HELPERS}
    sources = {}
    for name, expected in frozen.items():
        sources[name] = archive.path(name).read_bytes()
        actual = hashlib.sha256(sources[name]).hexdigest()
        require(actual == expected, f"Frozen helper hash changed: {name}")
        archive.hashes[name] = actual

    def module(name, filename):
        result = types.ModuleType(name)
        result.__file__ = str(archive.path(filename))
        exec(compile(sources[filename], result.__file__, "exec"), result.__dict__)
        return result

    previous = sys.modules.get("campaign_controls")
    bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.modules["campaign_controls"] = module("campaign_controls", "campaign_controls.py")
        yield module("batch_campaign_offline", "batch-campaign.py")
    finally:
        sys.dont_write_bytecode = bytecode
        if previous is None:
            sys.modules.pop("campaign_controls", None)
        else:
            sys.modules["campaign_controls"] = previous


def validate_campaign(campaign):
    archive = Archive(campaign)
    status = archive.read("batch-status.json")
    require(status.get("state") == status.get("phase") == "complete"
            and status.get("source_unchanged") is True
            and status.get("finished") is not None
            and status.get("child_pid") is None
            and status.get("current") is None
            and status.get("qualifications_passed") == 36
            and status.get("timing_workers") == 54,
            "Campaign is unfinished, failed, or lacks completion evidence")
    remote = PurePosixPath(status["cwd"])
    require(remote.is_absolute(), "Missing declared campaign directory")
    with frozen_controller(archive) as controller:
        frozen = {name: archive.hashes[name] for name in HELPERS}
        require(archive.read("queued-inputs.json") == frozen,
                "Queued helper hashes changed")
        provenance = archive.read("provenance.json")
        require(provenance["helper_sha256"] == {
            str(remote / name): sha for name, sha in frozen.items()
        }, "Provenance helper hashes changed")
        require(provenance["source"]["commit"] == controller.REVISION
                and not provenance["source"]["status"]
                and provenance["native"] == archive.read("native-provenance.json"),
                "Source/native provenance mismatch")
        require(archive.read("batch-plan.json") == provenance["plan"] == controller.PLAN,
                "Campaign plan differs from frozen controller")
        require(controller.REVISION == SCOPE["source_revision"]
                and controller.PLAN["qualification_seeds"] == [7102, 7103]
                and tuple(controller.BATCHES) == (1, 8, 32, 128, 512, 1024)
                and tuple(controller.ROUTES) == ("branch_standard", "torch_cpu", "torch_cuda")
                and all(controller.PLAN[key] == value for key, value in {
                    "bank_templates": 1024, "fft_samples": 131072, "blocks": 3,
                    "replicates": 3, "warmups": 2, "samples": 5, "seed": 7102,
                    "numerical_policy_version": 2}.items()),
                "Unsupported campaign scope or dimensions")
        decision = archive.read("policy-decision.json")
        require(decision.get("adopted") is True and decision.get("policy_version") == 2
                and decision.get("policy_document_sha256") == frozen["NUMERICAL-POLICY.md"],
                "Numerical policy was not adopted or its document changed")
        qualifications = {}
        expected_labels = set()
        timing_pids = set()
        intervals = []

        def check(label, result, route, batch, mode, seed, smoke=False):
            expected_labels.add(label)
            raw_name = f"runs/{label}/result.json"
            require(archive.read(raw_name) == result,
                    f"Aggregate differs from raw result: {label}")
            acquisition = archive.read(f"runs/{label}/acquisition.json")
            stage_name = f"stages/{label}.json"
            stage = archive.read(stage_name)
            require(acquisition["result_sha256"] == archive.hashes[raw_name]
                    and acquisition["stage_receipt"] == str(remote / stage_name),
                    f"Acquisition identity mismatch: {label}")
            require(stage["state"] == "complete" and stage["returncode"] == 0
                    and stage["name"] == label
                    and stage["pid"] == result["pid"] and stage["host"] == result["host"]
                    and result["host"] == status["host"],
                    f"Incomplete worker receipt: {label}")
            started, finished = stage["started"], stage["finished"]
            require(all(type(value) in (int, float) and math.isfinite(value)
                        for value in (started, finished)) and started <= finished,
                    f"Invalid worker interval: {label}")
            intervals.append((started, finished, mode, label))
            declared_output = str(remote / "runs" / label)
            require(result["arguments"]["output_dir"] == declared_output,
                    f"Worker output identity mismatch: {label}")
            phase = "smoke" if smoke else "qual"
            ref_label = f"{phase}-s{seed}-b1-branch_standard"
            declared_ref = str(remote / "runs" / ref_label)
            reference = None
            if label != ref_label:
                require(result["reference_directory"] == declared_ref
                        and result["arguments"]["reference_dir"] == declared_ref,
                        f"Reference directory mismatch: {label}")
                reference = OfflineReference(declared_ref, archive.path(f"runs/{ref_label}"))
            else:
                require(result["reference"]["directory"] == declared_output
                        and result["oracle"]["directory"] == declared_output,
                        f"Reference origin mismatch: {label}")
            controller.require_qualification(result)
            controller.validate_result(result, route, batch, mode, reference, smoke, seed)
            require(result["input_sha256"] == controller.canonical_hash(result["inputs"]),
                    f"Input identity hash mismatch: {label}")
            if route == "torch_cuda":
                require(result["runtime"]["device"] == "cuda:0"
                        and result["arguments"]["cuda_device"] == 0,
                        "CUDA evidence does not describe one GPU at cuda:0")
            if mode == "timing":
                require(result["timing"]["call_surface"] == "LiveBatchMatchedFilter.process_data",
                        "Timing does not use the public API")
                require(result["pid"] not in timing_pids, "Repeated timing worker PID")
                timing_pids.add(result["pid"])

        for smoke, phase, batches in ((True, "smoke", (1, 8)),
                                      (False, "qual", controller.BATCHES)):
            results = archive.read(f"{phase}-qualifications.json")
            controller.require_matrix(results, phase, batches)
            for seed in controller.PLAN["qualification_seeds"]:
                for batch in batches:
                    for route in controller.ROUTES:
                        label = f"{phase}-s{seed}-b{batch}-{route}"
                        check(label, results[label], route, batch, "qualify", seed, smoke)
            if not smoke:
                qualifications = results
        timings = archive.read("timings.json")
        # summarize checks the exact 54-cell replicate matrix as well as using
        # worker medians. Validation below must also pass before returning it.
        summary = controller.summarize(timings)
        for item in timings:
            route, batch = item["route"], item["batch"]
            check(f"time-r{item['repeat']}-b{batch}-{route}", item["result"],
                  route, batch, "timing", controller.PLAN["seed"])
        require(archive.read("summary.json") == summary,
                "Stored summary differs from recomputed controller summary")
        completed = status["completed"]
        require(len(completed) == len(expected_labels) and set(completed) == expected_labels,
                "Incomplete or duplicate completed-worker list")
        require(min(start for start, _, mode, _ in intervals if mode == "timing")
                >= max(end for _, end, mode, _ in intervals if mode == "qualify"),
                "Timing started before all qualifications finished")
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:]):
            require(previous[1] <= current[0],
                    f"Overlapping worker intervals: {previous[3]} and {current[3]}")
        accuracy = []
        for batch in controller.BATCHES:
            for route in controller.ROUTES:
                values = {key: max(
                    row[key]["max_abs_error"]
                    for seed in controller.PLAN["qualification_seeds"]
                    for block in qualifications[f"qual-s{seed}-b{batch}-{route}"]["pointwise_blocks"]
                    for row in block["rows"])
                    for key in ("normalized_oracle", "normalized_mkl_compatibility")}
                accuracy.append(dict(batch=batch, route=route, **values))
    archive.unchanged()
    return archive, summary, accuracy


def render(summary, accuracy, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batches, routes = summary["plan"]["batches"], summary["plan"]["routes"]
    caption = ("Shared host len; one numerical thread, CPU 8; CUDA additionally uses one GPU. No reservation.\n"
               "Source 9578a710479b (before squared-norm change); synthetic templates and strain; API timings uninstrumented.\n"
               "Excludes frame I/O, PSD, waveform/bank preparation and CLI overhead; not full-machine capacity.")
    with plt.rc_context({"font.size": 10, "svg.hashsalt": "pycbc-r4-batch"}):
        for name in ("throughput", "accuracy"):
            is_rate = name == "throughput"
            fig, axes = plt.subplots(1, 1 if is_rate else 2, figsize=(12, 6), squeeze=False)
            fig.subplots_adjust(left=0.09, right=0.97, bottom=0.30, top=0.77, wspace=0.27)
            fig.suptitle("Warm live-filter throughput" if is_rate else
                         "Live-filter numerical qualification", y=0.96, fontsize=18)
            subtitle = ("1,024 templates × 3 blocks; FFT 131072; median of 3 worker medians (5 timed iterations each).\n"
                        "Whiskers: observed min–max across the 3 workers; not a confidence interval."
                        if is_rate else
                        "Maximum over every sample, row and block of both full seeds 7102/7103.\n"
                        "Actual route normalization included; all 12 smoke and 36 full qualifications passed.")
            fig.text(0.5, 0.885, subtitle, ha="center", va="top", fontsize=10)
            for index, ax in enumerate(axes.flat):
                metric = ("normalized_oracle", "normalized_mkl_compatibility")[index]
                for route, label, color in zip(routes, LABELS, COLORS):
                    cells = [next(r for r in (summary["cells"] if is_rate else accuracy)
                                  if r["batch"] == batch and r["route"] == route) for batch in batches]
                    if is_rate:
                        values = [r["median_templates_per_second"] for r in cells]
                        errors = [[v - r["templates_per_second_range"][0] for v, r in zip(values, cells)],
                                  [r["templates_per_second_range"][1] - v for v, r in zip(values, cells)]]
                        ax.errorbar(range(6), values, yerr=errors, marker="o", capsize=4,
                                    color=color, label=label)
                    else:
                        ax.plot(range(6), [r[metric] for r in cells], marker="o", color=color, label=label)
                ax.set_xticks(range(6), batches)
                ax.set_xlabel("Execution batch size (templates)")
                ax.grid(alpha=0.2)
                ax.set_ylim(bottom=0)
                if is_rate:
                    ax.set_ylabel("Template-block evaluations / second")
                    ax.legend(fontsize=9)
                else:
                    ax.axhline(0.001, color="#8B2635", linestyle="--", label="v2 threshold: 0.001")
                    ax.set_yscale("symlog", linthresh=1e-8)
                    ax.set_ylim(0, 0.0012)
                    ax.set_title(("Independent complex128 oracle", "Standard CPU / MKL compatibility")[index])
                    ax.set_ylabel("Max |Δ complex SNR|")
            if not is_rate:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.18), ncol=4, fontsize=9)
            plot_caption = caption if is_rate else (
                "Accuracy axis: symlog; linear near zero below 1e-8. Exact zero is shown at 0.\n" + caption)
            fig.text(0.06, 0.04, plot_caption, fontsize=9, linespacing=1.5)
            for extension in ("png", "svg"):
                metadata = {"Date": None} if extension == "svg" else {}
                path = output / f"batch-{name}.{extension}"
                fig.savefig(path, dpi=160, metadata=metadata)
                if extension == "svg":
                    path.write_bytes(b"\n".join(
                        line.rstrip() for line in path.read_bytes().splitlines()) + b"\n")
            plt.close(fig)
    return matplotlib.__version__


def run(campaign, output, verify_only=False):
    output = Path(output).resolve()
    root = Path(campaign).resolve()
    require(not output.is_relative_to(root) and not root.is_relative_to(output),
            "Output and campaign directories must be separate")
    archive, summary, accuracy = validate_campaign(root)
    identity = {"schema_version": 1, "input_sha256": archive.hashes,
                "renderer_sha256": digest(Path(__file__).resolve()), "scope": SCOPE}
    manifest_path = output / "manifest.json"
    if verify_only:
        manifest = read_json(manifest_path.read_bytes())
        require(all(manifest.get(key) == value for key, value in identity.items()),
                "Manifest input, renderer, or scope mismatch")
        require(set(manifest["image_sha256"]) == set(IMAGES), "Incomplete image manifest")
        for name, expected in manifest["image_sha256"].items():
            require(digest(output / name) == expected, f"Image hash mismatch: {name}")
    else:
        require(not output.exists() or not any(output.iterdir()),
                "Output directory must be new or empty; use --verify-only for existing plots")
        output.mkdir(parents=True, exist_ok=True)
        version = render(summary, accuracy, output)
        archive.unchanged()
        manifest = dict(identity, matplotlib_version=version,
                        image_sha256={name: digest(output / name) for name in IMAGES})
        with manifest_path.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args.campaign, args.output, args.verify_only)
    except (ValueError, KeyError, TypeError, OSError, ZeroDivisionError) as exc:
        parser.exit(1, f"Batch plots rejected: {exc}\n")
    print("Verified batch plots" if args.verify_only else f"Rendered batch plots: {args.output}")


if __name__ == "__main__":
    main()
