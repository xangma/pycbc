#!/usr/bin/env python3
"""Bounded actual pycbc_live MPI smoke; execute this only on the compute host.

The worker observes provider and filter entry points without changing results.
This is dispatch/conditioning/filtering evidence, not a parity or speed claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def worker():
    from mpi4py import MPI
    import torch
    import pycbc
    import torchwave
    from pycbc.waveform import torchwave as provider
    from pycbc.filter.gpu_search import TiledLiveBatchMatchedFilter

    rank = MPI.COMM_WORLD.Get_rank()
    receipt = {"rank": rank, "host": platform.node(), "pid": os.getpid(),
               "pycbc_file": pycbc.__file__, "torchwave_file": torchwave.__file__,
               "torch_version": torch.__version__,
               "provider_batches": [], "filter_calls": [], "completed": False}
    path = Path(os.environ["LIVE_SMOKE_OUTPUT"]) / f"rank-{rank}.json"
    generate = provider.generate_batch
    process = TiledLiveBatchMatchedFilter.process_data

    def observed_generate(*args, **kwargs):
        data, templates = generate(*args, **kwargs)
        receipt["provider_batches"].append({
            "shape": list(data.shape), "device": str(data.device),
            "dtype": str(data.dtype),
            "providers": [t.waveform_provider for t in templates],
            "ids": [int(t.id) for t in templates],
            "delta_f": [float(t.delta_f) for t in templates],
        })
        write_json(path, receipt)
        return data, templates

    def observed_process(self, reader):
        result = process(self, reader)
        receipt["filter_calls"].append({
            "detector": reader.detector, "end_time": float(reader.end_time),
            "device": str(self.device),
            "triggers": len(result["snr"]) if isinstance(result, dict) else 0,
        })
        write_json(path, receipt)
        return result

    provider.generate_batch = observed_generate
    TiledLiveBatchMatchedFilter.process_data = observed_process
    repo = Path(os.environ["LIVE_SMOKE_REPO"])
    sys.argv = [str(repo / "bin/pycbc_live"), *sys.argv[2:]]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        receipt["completed"] = True
    except BaseException:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        write_json(path, receipt)


def prepare(output):
    import h5py
    import numpy as np
    from pycbc.frame import write_frame
    from pycbc.types import TimeSeries

    start, frame_duration, sample_rate = 1272790000, 64, 1024
    bank_path = output / "bank.hdf"
    with h5py.File(bank_path, "w") as bank:
        bank["mass1"] = [24., 20.]
        bank["mass2"] = [18., 15.]
        bank["spin1z"] = [.2, -.1]
        bank["spin2z"] = [-.1, .1]
        bank["f_lower"] = [30., 30.]
        bank["f_final"] = [256., 256.]
        bank.attrs["parameters"] = list(bank)
    frames = {}
    for ifo, seed in (("H1", 731), ("L1", 732)):
        noise = np.random.default_rng(seed).normal(
            scale=1e-21, size=frame_duration * sample_rate)
        series = TimeSeries(noise, delta_t=1.0 / sample_rate, epoch=start)
        frame = output / f"{ifo}-SMOKE-{start}-{frame_duration}.gwf"
        write_frame(str(frame), f"{ifo}:SMOKE", series)
        frames[ifo] = frame
    return bank_path, frames, start, sample_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--snr-threshold", type=float, default=1e6)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("use a new empty output directory to preserve prior receipts")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    cli = repo / "bin/pycbc_live"
    if "--enable-torchwave" not in cli.read_text():
        parser.error("repo must contain the integrated Live CLI provider flag")
    if not shutil.which(args.mpiexec):
        parser.error("MPI launcher unavailable: " + args.mpiexec)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    bank, frames, start, sample_rate = prepare(output)
    live_args = [
        "--bank-file", str(bank), "--sample-rate", str(sample_rate),
        "--approximant", "TaylorF2", "--enable-torchwave",
        "--low-frequency-cutoff", "30", "--max-length", "32",
        "--chisq-bins", "8", "--snr-threshold", str(args.snr_threshold),
        "--snr-abort-threshold", "1e9", "--newsnr-threshold", "0",
        "--max-triggers-in-batch", "4", "--analysis-chunk", "4",
        "--highpass-frequency", "20", "--highpass-bandwidth", "5",
        "--highpass-reduction", "60", "--psd-samples", "5",
        "--psd-segment-length", "2", "--psd-inverse-length", "1",
        "--psd-abort-difference", "100", "--psd-recalculate-difference", "100",
        "--trim-padding", ".25", "--frame-read-timeout", "1",
        "--channel-name", "H1:SMOKE", "L1:SMOKE",
        "--frame-src", f"H1:{frames['H1']}", f"L1:{frames['L1']}",
        "--processing-scheme", f"torch:{args.device}",
        "--fftw-measure-level", "0", "--increment", "4",
        "--max-batch-size", "65536", "--output-path", str(output / "triggers"),
        "--ranking-statistic", "quadsum", "--sngl-ranking", "newsnr",
        "--enable-background-estimation", "--background-ifar-limit", ".001",
        "--timeslide-interval", ".1", "--pvalue-lookback-time", "4",
        "--pvalue-combination-livetime", ".0005",
        "--ifar-double-followup-threshold", "1e9",
        "--ifar-upload-threshold", "1e9", "--start-time", str(start),
        "--end-time", str(start + 32), "--src-class-mchirp-to-delta", ".01",
        "--src-class-eff-to-lum-distance", ".74899",
        "--src-class-lum-distance-to-delta", "-.51557", "-.32195", "--verbose",
    ]
    command = [args.mpiexec, "-n", "2", args.python, "-m", "mpi4py",
               str(Path(__file__).resolve()), "--worker", *live_args]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", HDF5_USE_FILE_LOCKING="FALSE",
               PYTHONUNBUFFERED="1", LIVE_SMOKE_REPO=str(repo),
               LIVE_SMOKE_OUTPUT=str(output))
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    manifest = {
        "host": platform.node(), "cwd": str(repo), "command": command,
        "display_command": shlex.join(command), "device": args.device,
        "timeout_seconds": args.timeout_seconds, "log": str(output / "live.log"),
        "fixture": {"start": start, "analysis_seconds": 32, "frame_seconds": 64,
                    "sample_rate": sample_rate, "templates": 2,
                    "noise": "independent Gaussian, std=1e-21, seeds H1=731/L1=732",
                    "external_data": False, "injections": False},
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "status": "prepared", "scope": "Live dispatch and filtering smoke",
    }
    write_json(output / "manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps(manifest), flush=True)
        return
    started = time.monotonic()
    with (output / "live.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        manifest.update(pid=proc.pid, stop_command=f"kill -TERM -- -{proc.pid}",
                        expected_next_check="inspect live.log and rank receipts in 30 seconds")
        write_json(output / "manifest.json", manifest)
        print(json.dumps(manifest), flush=True)
        try:
            returncode = proc.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            returncode = 124
    receipts = [json.loads(p.read_text()) for p in sorted(output.glob("rank-*.json"))]
    workers = [r for r in receipts if r["rank"] > 0]
    batches = [b for r in workers for b in r["provider_batches"]]
    calls = [c for r in workers for c in r["filter_calls"]]
    expected_device = args.device
    checks = {
        "cli_exit_zero": returncode == 0,
        "both_ranks_completed": len(receipts) == 2 and all(r["completed"] for r in receipts),
        "native_templates_generated": bool(batches) and all(
            b["providers"] and set(b["providers"]) == {"torchwave"} for b in batches),
        "provider_device": bool(batches) and all(
            b["device"].split(":")[0] == expected_device for b in batches),
        "both_detectors_filtered": {c["detector"] for c in calls} == {"H1", "L1"},
        "filter_device": bool(calls) and all(
            c["device"].split(":")[0] == expected_device for c in calls),
        "hdf_outputs_written": bool(list((output / "triggers").rglob("*.hdf"))),
    }
    manifest.update(returncode=returncode, wall_seconds=time.monotonic() - started,
                    checks=checks, status="passed" if all(checks.values()) else "failed")
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
    else:
        main()
