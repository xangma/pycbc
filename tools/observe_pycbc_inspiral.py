#!/usr/bin/env python
"""Run an upstream ``pycbc_inspiral`` with opt-in observation snapshots.

The upstream program is read and compiled in memory.  Two statements are
inserted at checked, semantic boundaries: after PSD association and after the
final trigger output.  The input script is never changed.
"""

import ast
import json
import os
import sys
from pathlib import Path


def _array(value):
    """Return a numpy-like value without importing PyCBC."""
    value = value.numpy() if hasattr(value, "numpy") else value
    return value


def _save_evidence(strain, segments):
    path = os.environ.get("PYCBC_BENCHMARK_EVIDENCE")
    if not path:
        return
    import h5py

    with h5py.File(path, "w") as out:
        out.attrs["science_config"] = os.environ.get(
            "PYCBC_BENCHMARK_SCIENCE_CONFIG", "{}")
        strain_data = _array(strain)
        out.create_dataset("conditioned_strain", data=strain_data)
        out.attrs["strain_epoch"] = float(strain.start_time)
        out.attrs["strain_delta_t"] = float(strain.delta_t)
        for index, segment in enumerate(segments):
            group = out.create_group("segments/%d" % index)
            segment_data = _array(segment)
            psd_data = _array(segment.psd)
            group.create_dataset("strain", data=segment_data)
            group.create_dataset("psd", data=psd_data)
            group.attrs["delta_f"] = float(segment.delta_f)
            group.attrs["epoch"] = float(segment.epoch)
            group.attrs["analyze_start"] = int(segment.analyze.start)
            group.attrs["analyze_stop"] = int(segment.analyze.stop)
            group.attrs["segment_start"] = int(segment.seg_slice.start)
            group.attrs["segment_stop"] = int(segment.seg_slice.stop)


def _save_work(segments, completed_templates, sample_rate):
    path = os.environ.get("PYCBC_BENCHMARK_WORK")
    if not path:
        return
    intervals = sorted(
        (int(segment.cumulative_index),
         int(segment.cumulative_index + segment.analyze.stop -
             segment.analyze.start))
        for segment in segments)
    merged = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    seconds = sum(stop - start for start, stop in merged) / sample_rate
    with open(path, "w") as output:
        json.dump({
            "completed_templates": int(completed_templates),
            "valid_detector_seconds": seconds,
            "valid_sample_intervals": merged,
            "sample_rate": sample_rate,
            "signal_dtypes": sorted({str(_array(segment).dtype)
                                      for segment in segments}),
            "completed_template_seconds": completed_templates * seconds,
        }, output)


class _Observer:
    def __init__(self):
        self._evidence_saved = False

    def evidence(self, strain, segments):
        # JAX and native upstream variants can reach this boundary more than
        # once while preparing the same segments.  Keep one complete snapshot.
        if not self._evidence_saved:
            _save_evidence(strain, segments)
            self._evidence_saved = True

    @staticmethod
    def work(segments, completed_templates, sample_rate):
        _save_work(segments, completed_templates, sample_rate)


def _is_associate(call):
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "associate_psds_to_segments"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "psd")


def _is_final_write(statement):
    return (isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "write_events"
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "event_mgr")


def _instrument(source, filename):
    tree = ast.parse(source, filename=filename)
    if any(isinstance(node, ast.Name) and node.id == '__pycbc_observer'
           for node in ast.walk(tree)):
        raise RuntimeError('refusing to observe: observer namespace already used')
    associates = []
    for node in ast.walk(tree):
        if _is_associate(node):
            associates.append(node)
    if len(associates) != 1:
        raise RuntimeError(
            "refusing to observe: expected exactly one upstream PSD "
            "association boundary, found %d" % len(associates))
    if not any(isinstance(node, ast.Expr) and node.value is associates[0]
               for node in ast.walk(tree)):
        raise RuntimeError('refusing to observe: unsupported PSD association boundary')

    # Only a module-level write is the successful completion boundary.  The
    # early injection-only exit is nested in an if statement and is excluded.
    final_writes = [node for node in tree.body if _is_final_write(node)]
    if len(final_writes) != 1:
        raise RuntimeError(
            "refusing to observe: expected exactly one final event output "
            "boundary, found %d" % len(final_writes))

    class Inject(ast.NodeTransformer):
        _final_node = final_writes[0]

        def visit_Expr(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Call) and _is_associate(node.value):
                call = ast.Expr(ast.Call(
                    func=ast.Attribute(ast.Name("__pycbc_observer", ast.Load()),
                                       "evidence", ast.Load()),
                    args=[ast.Name("gwstrain", ast.Load()),
                          ast.Name("segments", ast.Load())], keywords=[]))
                return [node, ast.copy_location(call, node)]
            if node is self._final_node:
                call = ast.Expr(ast.Call(
                    func=ast.Attribute(ast.Name("__pycbc_observer", ast.Load()),
                                       "work", ast.Load()),
                    args=[ast.Name("segments", ast.Load()),
                          ast.Call(ast.Name("len", ast.Load()),
                                   [ast.Name("tanalyze", ast.Load())], []),
                          ast.Call(ast.Name("float", ast.Load()),
                                   [ast.Attribute(ast.Name("opt", ast.Load()),
                                                  "sample_rate", ast.Load())],
                                   [])], keywords=[]))
                return [node, ast.copy_location(call, node)]
            return node

    tree = Inject().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def validate_source(path):
    """Fail closed if *path* is not the expected pristine script shape."""
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError("original pycbc_inspiral not found: %s" % path)
    _instrument(path.read_text(), str(path))
    return path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: observe_pycbc_inspiral.py ORIGINAL_PYCBC_INSPIRAL [ARGS...]")
        return 0 if argv else 2
    script = validate_source(argv[0])
    source = script.read_text()
    tree = _instrument(source, str(script))
    sys.argv = [str(script)] + argv[1:]
    # Match direct execution of the upstream executable, whose import search
    # path starts beside the script rather than beside this wrapper.
    if sys.path:
        sys.path[0] = str(script.parent)
    else:
        sys.path.insert(0, str(script.parent))
    namespace = {
        "__name__": "__main__", "__file__": str(script),
        "__package__": None, "__cached__": None,
        "__pycbc_observer": _Observer(),
    }
    exec(compile(tree, str(script), "exec"), namespace, namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
