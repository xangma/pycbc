"""External public-API inference benchmark; never edits the target checkout."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

EXPECTED_SHA = "607bce53ead14f12af32552a5b2441d3bc667267"
SCRIPT = Path(__file__).resolve()
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(v) for v in value]
        return value
    Path(path).write_text(json.dumps(finite(obj), indent=2, allow_nan=False) + "\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def provenance(args):
    sha = git(args.root, "rev-parse", "HEAD")
    status = git(args.root, "status", "--porcelain", "--untracked-files=normal")
    result = dict(sha=sha, clean=not status, status=status, root=str(args.root),
                  python=sys.executable, python_version=sys.version, host=platform.node(),
                  platform=platform.platform(), pid=os.getpid(), argv=sys.argv,
                  harness_sha256=digest(SCRIPT), threads=args.threads,
                  thread_environment={v: os.environ.get(v) for v in THREAD_VARS})
    if sha != EXPECTED_SHA or status:
        raise RuntimeError(f"Expected clean {EXPECTED_SHA}; got {sha}, status={status!r}")
    return result


def imports(args):
    sys.path.insert(0, str(args.root))
    import numpy as np
    import pycbc
    from pycbc import scheme
    from pycbc.types import FrequencySeries
    from pycbc.inference.models.gaussian_noise import GaussianNoise
    from pycbc.inference.models.relbin import Relative
    assert Path(pycbc.__file__).resolve().is_relative_to(args.root)
    return np, scheme, FrequencySeries, {"gaussian": GaussianNoise, "relative": Relative}


def make_model(args, case, arrays, lib):
    np, scheme, FrequencySeries, classes = lib
    data = {d: FrequencySeries(arrays["data_" + d].copy(), delta_f=case["delta_f"],
                              epoch=case["epoch"]) for d in case["detectors"]}
    psds = {d: FrequencySeries(arrays["psd_" + d].copy(), delta_f=case["delta_f"])
            for d in case["detectors"]}
    kwargs = dict(psds=psds, static_params=case["static"].copy(),
                  high_frequency_cutoff={d: case["f_high"] for d in data},
                  normalize=False, ignore_failed_waveforms=False)
    if args.model == "relative":
        kwargs.update(fiducial_params=case["injection"].copy(), epsilon=0.1,
                      marginalize_phase=False, marginalize_distance=False,
                      earth_rotation=False)
    return classes[args.model](tuple(case["variable_params"]), data,
                              {d: case["f_low"] for d in data}, **kwargs)


def evaluate(model, params):
    model.update(**params)
    return float(model.loglikelihood)


def prepare(args):
    result = provenance(args)
    lib = imports(args)
    np, scheme, FrequencySeries, classes = lib
    from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator
    case = dict(seed=20260906, epoch=1126259450.0, duration=args.duration,
                sample_rate=args.sample_rate, delta_f=1.0 / args.duration,
                detectors=["H1", "L1"], f_low=20.0, f_high=args.sample_rate / 2 - 1,
                precision="float64/complex128", groups=args.groups,
                evaluations_per_group=args.evaluations,
                static=dict(approximant="TaylorF2", f_lower=20.0,
                            spin1z=0.0, spin2z=0.0),
                injection=dict(mass1=10.0, mass2=8.0, distance=500.0,
                               inclination=0.4, coa_phase=0.2, ra=1.1, dec=-0.3,
                               polarization=0.2, tc=1126259450.0 + 0.8 * args.duration))
    case["variable_params"] = list(case["injection"])
    rng = np.random.default_rng(case["seed"])
    widths = dict(mass1=0.015, mass2=0.015, distance=2.0, inclination=0.003,
                  coa_phase=0.006, ra=0.003, dec=0.003, polarization=0.004, tc=0.0003)
    points = [{k: v + widths[k] * float(rng.uniform(-1, 1))
               for k, v in case["injection"].items()}
              for _ in range(1 + 12 + 8 + args.groups * args.evaluations)]
    assert len({json.dumps(p, sort_keys=True) for p in points}) == len(points)
    assert len({(p["mass1"], p["mass2"]) for p in points}) == len(points)
    case.update(cold_point=points[0], parity_points=points[1:13],
                warmup_points=points[13:21], timing_points=points[21:])
    count = int(args.duration * args.sample_rate) // 2 + 1
    case["frequency_bins"] = count
    case["psd_formula"] = "1e-46 * ((40/max(f,1))**4 + 2 + (f/300)**2)"
    case["noise_formula"] = "sqrt(PSD/(4*delta_f))*(N(0,1)+i*N(0,1)); DC/Nyquist=0"
    with scheme.CPUScheme(num_threads=args.threads):
        generator = FDomainDetFrameGenerator(
            FDomainCBCGenerator, case["epoch"], detectors=case["detectors"],
            variable_args=tuple(case["injection"]), delta_f=case["delta_f"],
            **case["static"])
        injection = generator.generate(**case["injection"])
        f = np.arange(count) * case["delta_f"]
        psd = 1e-46 * ((40 / np.maximum(f, 1)) ** 4 + 2 + (f / 300) ** 2)
        arrays = {}
        for detector in case["detectors"]:
            signal = injection[detector].copy()
            signal.resize(count)
            noise = np.sqrt(psd / (4 * case["delta_f"])) * (
                rng.normal(size=count) + 1j * rng.normal(size=count))
            noise[[0, -1]] = 0
            arrays["data_" + detector] = np.asarray(signal.numpy(), dtype=np.complex128) + noise
            arrays["psd_" + detector] = psd.copy()
        np.savez(args.output / "case.npz", **arrays)
        reference = {}
        for model_name in ("gaussian", "relative"):
            args.model = model_name
            try:
                model = make_model(args, case, arrays, lib)
                values = []
                for point in case["parity_points"]:
                    likelihood = evaluate(model, point)
                    values.append(dict(loglikelihood=likelihood, loglr=float(model.loglr)))
                if not np.isfinite([[v["loglikelihood"], v["loglr"]] for v in values]).all():
                    raise RuntimeError("Non-finite reference likelihood")
                reference[model_name] = dict(status="ok", values=values,
                                            model_class=type(model).__module__ + "." + type(model).__name__)
            except Exception as exc:
                reference[model_name] = dict(status="unsupported", error=repr(exc),
                                            traceback=traceback.format_exc())
    case.update(reference=reference, preparation=result, data_sha256=digest(args.output / "case.npz"))
    write(args.output / "case.json", case)
    return dict(status="prepared", reference=reference, case=str(args.output / "case.json"))


def worker(args):
    result = provenance(args)
    partial_path = args.output / f"{args.model}-{args.route}-t{args.threads}-r{args.replicate}.json"
    case_path = args.output / "case.json"
    case = json.loads(case_path.read_text())
    assert case["preparation"]["sha"] == result["sha"]
    assert digest(args.output / "case.npz") == case["data_sha256"]
    result.update(route=args.route, replicate=args.replicate, model=args.model,
                  case_sha256=digest(case_path), data_sha256=case["data_sha256"],
                  status="initializing", group_samples=[], parity=[],
                  timing_boundary="Public model.update(**varied_parameters) + float(model.loglikelihood); CUDA synchronized before/after every timed group. Includes waveform generation, projection, likelihood reduction and host scalar return. Excludes setup, parity and warmup.")
    write(partial_path, result)
    start = time.perf_counter_ns()
    lib = imports(args)
    result["imports_ns"] = time.perf_counter_ns() - start
    write(partial_path, result)
    np, scheme, _, _ = lib
    result["numpy_version"] = np.__version__
    try:
        from threadpoolctl import threadpool_info
        result["threadpools"] = threadpool_info()
    except ImportError:
        result["threadpools"] = "threadpoolctl unavailable"
    arrays = dict(np.load(args.output / "case.npz"))
    if case["reference"][args.model]["status"] != "ok":
        result.update(status="unsupported_reference", reference=case["reference"][args.model])
        return result
    torch = None
    if args.route != "standard_cpu":
        import torch
        result.update(torch_version=torch.__version__, cuda_build=torch.version.cuda)
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        if args.route == "cuda" and not torch.cuda.is_available():
            result.update(status="unsupported", reason="torch.cuda.is_available() is false")
            return result
    def synchronize():
        if args.route == "cuda":
            torch.cuda.synchronize()
    start = time.perf_counter_ns()
    context = (scheme.CPUScheme(num_threads=args.threads) if args.route == "standard_cpu"
               else scheme.TorchScheme("cuda" if args.route == "cuda" else "cpu",
                                       num_threads=args.threads))
    with context:
        synchronize()
        result["cold_scheme_ns"] = time.perf_counter_ns() - start
        if args.route == "cuda":
            result["device"] = torch.cuda.get_device_name()
        if torch is not None:
            result["torch_threads"] = torch.get_num_threads()
            result["torch_interop_threads"] = torch.get_num_interop_threads()
        start = time.perf_counter_ns()
        model = make_model(args, case, arrays, lib)
        synchronize()
        result["cold_model_setup_ns"] = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        cold = evaluate(model, case["cold_point"])
        synchronize()
        result["cold_first_update_likelihood_ns"] = time.perf_counter_ns() - start
        if not np.isfinite(cold):
            raise RuntimeError("Non-finite cold likelihood")
        result["cold_loglikelihood"] = cold
        raw_likelihood = model.loglikelihood
        result["likelihood_result"] = dict(type=type(raw_likelihood).__module__ + "." + type(raw_likelihood).__name__,
                                            device=str(getattr(raw_likelihood, "device", "cpu")))
        result["model_class"] = type(model).__module__ + "." + type(model).__name__
        result["data_storage"] = {d: dict(type=type(v.data).__module__ + "." + type(v.data).__name__,
                                            device=str(getattr(v.data, "device", "cpu")), dtype=str(v.dtype))
                                  for d, v in model.data.items()}
        if args.model == "relative":
            result["relative_bins"] = {d: len(v) - 1 for d, v in model.edges.items()}
        write(partial_path, result)
        for point, expected in zip(case["parity_points"], case["reference"][args.model]["values"], strict=True):
            actual = dict(loglikelihood=evaluate(model, point), loglr=float(model.loglr))
            row = dict(params=point, actual=actual, expected=expected)
            row["absolute_errors"] = {k: abs(actual[k] - expected[k]) for k in actual}
            row["passed"] = all(np.isclose(actual[k], expected[k], rtol=args.rtol, atol=args.atol) for k in actual)
            result["parity"].append(row)
        result["parity_tolerance"] = dict(rtol=args.rtol, atol=args.atol)
        write(partial_path, result)
        if not all(p["passed"] for p in result["parity"]):
            result["status"] = "parity_failed"
            return result
        for point in case["warmup_points"]:
            evaluate(model, point)
        synchronize()
        width = case["evaluations_per_group"]
        for group in range(case["groups"]):
            points = case["timing_points"][group * width:(group + 1) * width]
            synchronize()
            start = time.perf_counter_ns()
            values = [evaluate(model, point) for point in points]
            synchronize()
            elapsed = time.perf_counter_ns() - start
            result["group_samples"].append(dict(group=group, total_ns=elapsed,
                                                evaluations=len(points), ns_per_evaluation=elapsed / len(points),
                                                loglikelihoods=values))
            write(partial_path, result)
            if not np.isfinite(values).all():
                raise RuntimeError("Non-finite timed likelihood")
    result.update(status="ok", final_sha=git(args.root, "rev-parse", "HEAD"),
                  final_status=git(args.root, "status", "--porcelain", "--untracked-files=normal"))
    assert result["final_sha"] == EXPECTED_SHA and not result["final_status"]
    return result


def orchestrate(args):
    commands = []
    common = [args.python, str(SCRIPT), "--root", str(args.root), "--output", str(args.output)]
    commands.append(("prepare", common + ["--mode", "prepare", "--threads", "1",
                     "--groups", str(args.groups), "--evaluations", str(args.evaluations),
                     "--duration", str(args.duration), "--sample-rate", str(args.sample_rate)]))
    for model in ("gaussian", "relative"):
        for route in ("standard_cpu", "torch_cpu", "cuda"):
            for threads in ([1, 4] if route != "cuda" else [1]):
                for replicate in range(3):
                    key = f"{model}-{route}-t{threads}-r{replicate}"
                    commands.append((key, common + ["--mode", "worker", "--model", model,
                                    "--route", route, "--threads", str(threads),
                                    "--replicate", str(replicate), "--rtol", str(args.rtol), "--atol", str(args.atol)]))
    write(args.output / "commands.json", commands)
    receipts = []
    for key, command in commands:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(args.root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        count = command[command.index("--threads") + 1]
        env.update({name: count for name in THREAD_VARS})
        start = time.perf_counter_ns()
        with (args.output / (key + ".log")).open("w") as log:
            process = subprocess.run(command, cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipts.append(dict(key=key, command=command, returncode=process.returncode,
                             subprocess_wall_ns=time.perf_counter_ns() - start))
        write(args.output / "process-receipts.json", receipts)
        print(key, process.returncode, flush=True)
        if key == "prepare" and process.returncode:
            break
    return dict(status="complete_with_failures" if any(r["returncode"] for r in receipts) else "complete",
                receipts=receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "worker", "orchestrate"), default="worker")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("standard_cpu", "torch_cpu", "cuda"), default="standard_cpu")
    parser.add_argument("--model", choices=("gaussian", "relative"), default="gaussian")
    parser.add_argument("--threads", type=int, choices=(1, 4), default=1)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--evaluations", type=int, default=16)
    parser.add_argument("--duration", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=2048)
    parser.add_argument("--rtol", type=float, default=2e-7)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.groups < 1 or args.evaluations < 1 or args.duration < 1 or args.sample_rate < 64:
        parser.error("groups/evaluations/duration must be positive; sample-rate must be >=64")
    for name in THREAD_VARS:
        os.environ[name] = str(args.threads)
    os.environ["PYTHONPATH"] = str(args.root) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    # The direct worker also prepends sys.path before the first PyCBC import.
    name = (f"{args.model}-{args.route}-t{args.threads}-r{args.replicate}" if args.mode == "worker" else args.mode)
    write(args.output / (name + ".json"), dict(status="starting", argv=sys.argv,
          harness_sha256=digest(SCRIPT), root=str(args.root), python=sys.executable,
          threads=args.threads, route=args.route, model=args.model, replicate=args.replicate))
    try:
        result = {"prepare": prepare, "worker": worker, "orchestrate": orchestrate}[args.mode](args)
    except Exception as exc:
        partial_path = args.output / (name + ".json")
        result = json.loads(partial_path.read_text()) if partial_path.exists() else {}
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc(),
                      argv=sys.argv, harness_sha256=digest(SCRIPT), host=platform.node(), pid=os.getpid(),
                      root=str(args.root), python=sys.executable, threads=args.threads,
                      route=args.route, model=args.model, replicate=args.replicate)
        try:
            result.update(sha=git(args.root, "rev-parse", "HEAD"),
                          final_status=git(args.root, "status", "--porcelain", "--untracked-files=normal"))
        except Exception:
            pass
    write(args.output / (name + ".json"), result)
    print(json.dumps({"result": name, "status": result["status"]}), flush=True)
    return 0 if result["status"] in ("ok", "prepared", "complete", "unsupported", "unsupported_reference") else 1


if __name__ == "__main__":
    raise SystemExit(main())
