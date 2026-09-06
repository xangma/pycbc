#!/usr/bin/env python3
"""Run strain/PSD regressions with the available local CPU extension ABI."""

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
        "types.array_cpu", "vetoes.chisq_cpu", "events.eventmgr_cython",
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

    from pycbc import psd
    from pycbc.strain import strain

    for module, relative, name, class_name in [
        (psd, "pycbc/psd/__init__.py", "from_cli", None),
        (strain, "pycbc/strain/strain.py", "fourier_segments", "StrainSegments"),
    ]:
        paths.append(worktree / relative)
        if args.baseline:
            source = subprocess.check_output(
                ["git", "show", "968bcd558117262af0d603710b054174659adb51:" + relative],
                cwd=worktree, text=True,
            )
            nodes = ast.parse(source).body
            if class_name:
                classes = [node for node in nodes if isinstance(node, ast.ClassDef)
                           and node.name == class_name]
                assert len(classes) == 1
                nodes = classes[0].body
            functions = [node for node in nodes if isinstance(node, ast.FunctionDef)
                         and node.name == name]
            assert len(functions) == 1
            baseline = ast.Module(body=functions, type_ignores=[])
            namespace = vars(module) if class_name is None else dict(vars(module))
            exec(compile(baseline, "baseline-968:" + relative, "exec"), namespace)
            if class_name:
                setattr(getattr(module, class_name), name, namespace[name])

    tests = ["test/test_strain_psd_precision.py"]
    if args.existing:
        assert not args.baseline
        tests = ["test/test_psd.py", "test/test_strain.py",
                 "test/test_torch_psd_pipeline.py", "test/test_torch_psd_protocol.py"]
    paths += [worktree / name for name in tests]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    print(json.dumps({
        "baseline_selected_functions_from_source": (
            "968bcd558117262af0d603710b054174659adb51" if args.baseline else None
        ),
        "local_native_abi": "Existing Darwin Python 3.12 extensions loaded read-only",
        "python": sys.executable, "tests": tests, "file_sha256": before,
    }, indent=2), flush=True)
    import pytest

    result = pytest.main(["-q", "-p", "no:cacheprovider", *tests])
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after, "Inputs changed during validation"
    return result


if __name__ == "__main__":
    raise SystemExit(main())
