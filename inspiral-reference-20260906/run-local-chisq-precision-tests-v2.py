#!/usr/bin/env python3
"""Run precision regressions with the available local CPU extension ABI."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--existing", action="store_true")
    args = parser.parse_args()
    worktree = Path("/private/tmp/pycbc-torch-performance-fix-20260906")
    sys.path.insert(0, str(worktree))
    importlib.import_module("pycbc.types")

    paths = [Path(__file__).resolve()]
    for name in [
        "types.array_cpu",
        "vetoes.chisq_cpu",
        "events.eventmgr_cython",
        "events.simd_threshold_cython",
    ]:
        path = Path("/Users/xangma/repos/pycbc/pycbc") / (
            name.replace(".", "/") + ".cpython-312-darwin.so"
        )
        spec = importlib.util.spec_from_file_location("pycbc." + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        paths.append(path)

    from pycbc.filter import matchedfilter
    from pycbc.vetoes import chisq

    replacements = [
        (matchedfilter, "pycbc/filter/matchedfilter.py", ["sigmasq_series"]),
        (
            chisq,
            "pycbc/vetoes/chisq.py",
            [
                "power_chisq_bins_from_sigmasq_series",
                "power_chisq_at_points_from_precomputed",
            ],
        ),
    ]
    for module, relative, names in replacements:
        paths.append(worktree / relative)
        if args.baseline:
            source = subprocess.check_output(
                [
                    "git",
                    "show",
                    "968bcd558117262af0d603710b054174659adb51:" + relative,
                ],
                cwd=worktree,
                text=True,
            )
            functions = [
                node
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name in names
            ]
            assert len(functions) == len(names)
            baseline = ast.Module(body=functions, type_ignores=[])
            exec(compile(baseline, "baseline-968:" + relative, "exec"), vars(module))
    tests = [
        "test/test_sigmasq_series_precision.py",
        "test/test_chisq_precision.py",
    ]
    if args.existing:
        assert not args.baseline
        tests = [
            "test/test_torch_chisq_cpu_optimization.py",
            "test/test_torch_chisq_sparse_dispatch.py",
            "test/test_torch_filter_pipeline.py::test_power_chisq_bins_stay_on_device",
            "test/test_torch_filter_pipeline.py::test_power_chisq_bins_accepts_public_backend_storage",
        ]
    paths += [worktree / name.split("::")[0] for name in tests]
    print(
        json.dumps(
            {
                "baseline_selected_functions_from_source": (
                    "968bcd558117262af0d603710b054174659adb51"
                    if args.baseline
                    else None
                ),
                "local_native_abi": "Existing Darwin Python 3.12 extensions loaded read-only",
                "python": sys.executable,
                "tests": tests,
                "file_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in paths
                },
            },
            indent=2,
        ),
        flush=True,
    )
    import pytest

    return pytest.main(["-q", "-p", "no:cacheprovider", *tests])


if __name__ == "__main__":
    raise SystemExit(main())
