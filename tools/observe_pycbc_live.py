#!/usr/bin/env python3
"""Observe a pristine pycbc_live script without replacing its calculations.

The AST additions only emit stage events, copy qualification evidence, and
pace cached input when explicitly requested. Original statements, function
calls, arrays and scientific arguments are preserved. No checkout is edited.
"""
import ast
import json
import math
import os
from pathlib import Path
import sys
import time
import types


def _call_name(node):
    return ast.unparse(node.func) if isinstance(node, ast.Call) else None


def instrument_source(source, filename="pycbc_live"):
    tree = ast.parse(source, filename)
    if any(isinstance(n, ast.Name) and n.id == "_benchmark_observer"
           for n in ast.walk(tree)):
        raise ValueError("observer namespace already used by executable")
    counts = dict(loop=0, read=0, filter=0, output=0, postprocess=0)

    def hook(method, *args):
        params = ", ".join(repr(a) for a in args)
        return ast.parse("_benchmark_observer.%s(locals()%s)" %
                         (method, ", " + params if params else "")).body[0]

    class AddObservations(ast.NodeTransformer):
        def visit_While(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.test) == "data_end() < args.end_time":
                counts["loop"] += 1
                node.body = [hook("start_block"), *node.body, hook("end_block")]
            return node

        def visit_Assign(self, node):
            name = _call_name(node.value)
            if name == "data_reader[ifo].advance":
                counts["read"] += 1
                return [hook("stage", "frame_read", "start"), node,
                        hook("stage", "frame_read", "end")]
            if name == "mf.process_data":
                counts["filter"] += 1
                return [hook("stage", "filter", "start"), node,
                        hook("stage", "filter", "end")]
            return node

        def visit_Expr(self, node):
            if _call_name(node.value) == "evnt.dump":
                counts["output"] += 1
                return [hook("stage", "output", "start"),
                        hook("save_evidence"), node, hook("stage", "output", "end")]
            return node

        def visit_If(self, node):
            node = self.generic_visit(node)
            if (ast.unparse(node.test) == "evnt.rank > 0" and node.orelse and
                    any(isinstance(n, ast.Call) and _call_name(n) == "evnt.commit_results"
                        for statement in node.body for n in ast.walk(statement))):
                counts["postprocess"] += 1
                node.orelse = [hook("stage", "postprocess", "start"), *node.orelse,
                              hook("stage", "postprocess", "end")]
            return node

    tree = AddObservations().visit(tree)
    if any(count != 1 for count in counts.values()):
        raise ValueError("unsupported upstream live observation boundaries: " + str(counts))
    return ast.fix_missing_locations(tree)


def validate_source(path):
    instrument_source(Path(path).read_text(), str(path))


class Observer:
    def __init__(self):
        self.mode = os.environ.get("PYCBC_OBSERVER_REPLAY_MODE", "unpaced")
        self.rate = float(os.environ.get("PYCBC_OBSERVER_REPLAY_RATE", "1"))
        if self.mode not in {"unpaced", "paced", "none"}:
            raise ValueError("invalid observer replay mode")
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("observer replay rate must be finite and positive")
        self.origin = self.wall_origin = None
        self.block_start = self.block_wall = None

    def emit(self, scope, name, event, **fields):
        stamp = time.perf_counter_ns()
        payload = dict(pid=os.getpid(), rank=scope["evnt"].rank,
                       name=name, stage=name, label=name, event=event,
                       time_ns=stamp, monotonic_ns=stamp,
                       data_start=self.block_start, **fields)
        print("PYCBC_STAGE_EVENT " + json.dumps(payload, separators=(",", ":")),
              file=sys.stderr, flush=True)

    def start_block(self, scope):
        self.block_start = float(scope["data_end"]())
        if self.origin is None:
            self.origin, self.wall_origin = self.block_start, time.perf_counter()
        if self.mode == "paced":
            delay = ((self.block_start - self.origin) / self.rate -
                     (time.perf_counter() - self.wall_origin))
            if delay > 0:
                time.sleep(delay)
        self.block_wall = time.perf_counter()
        self.emit(scope, "block", "start", analysis_chunk=float(scope["args"].analysis_chunk))

    def end_block(self, scope):
        elapsed = time.perf_counter() - self.block_wall
        end = float(scope["data_end"]())
        lag = (max(0.0, (time.perf_counter() - self.wall_origin) * self.rate -
                   (end - self.origin)) if self.mode == "paced" else 0.0)
        budget = float(scope["args"].analysis_chunk)
        if self.mode == "paced":
            budget /= self.rate
        self.emit(scope, "block", "end", data_end=end, elapsed_sec=elapsed,
                  processing_elapsed_sec=elapsed, lag_sec=lag,
                  live_detectors=sorted(scope["evnt"].live_detectors),
                  deadline_met=elapsed <= budget)

    def stage(self, scope, name, event):
        fields = {}
        if name in {"frame_read", "filter"}:
            fields["ifo"] = scope["ifo"]
        if name == "filter" and event == "end":
            mf = scope["mf"]
            fields["signal_dtypes"] = sorted({
                str(buffer.dtype) for buffers in (mf.out_mem, mf.cout_mem)
                for buffer in buffers.values()})
            fields["sample_rate"] = float(
                scope["data_reader"][scope["ifo"]].strain.sample_rate)
        self.emit(scope, name, event, **fields)

    def save_evidence(self, scope):
        path = os.environ.get("PYCBC_OBSERVER_EVIDENCE")
        if not path or scope["evnt"].rank != 0:
            return
        import h5py
        import numpy as np

        end = float(scope["data_end"]())
        Path(path).mkdir(parents=True, exist_ok=True)
        filename = Path(path) / f"block-{self.block_start:.6f}-{end:.6f}.hdf"
        with h5py.File(filename, "w") as evidence:
            evidence.attrs["science_config"] = os.environ["PYCBC_SCIENCE_CONFIG"]
            evidence.attrs["analyze_start"] = self.block_start
            evidence.attrs["analyze_end"] = end
            first_strain = True
            for ifo, reader in sorted(scope["data_reader"].items()):
                segment = evidence.create_group(f"segments/{ifo}")
                segment.attrs["analyze_start"] = self.block_start
                segment.attrs["analyze_end"] = end
                if reader.strain is not None:
                    segment.attrs["delta_t"] = float(reader.strain.delta_t)
                    segment.attrs["epoch"] = float(reader.strain.start_time)
                    data = np.asarray(reader.strain)
                    segment.create_dataset("strain", data=data)
                    if first_strain:
                        evidence.create_dataset("conditioned_strain", data=data)
                        first_strain = False
                psd = scope["psds"].get(ifo)
                if psd is not None:
                    segment.attrs["delta_f"] = float(psd.delta_f)
                    segment.create_dataset("psd", data=np.asarray(psd))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: observe_pycbc_live.py /reference/bin/pycbc_live [arguments]")
    executable = Path(argv[0]).resolve()
    tree = instrument_source(executable.read_text(), str(executable))
    # The original executable retains its native argv, imports and __file__.
    original_main, original_argv, original_path = sys.modules["__main__"], sys.argv, sys.path[0]
    module = types.ModuleType("__main__")
    module.__dict__.update(__file__=str(executable), __package__=None,
                           _benchmark_observer=Observer())
    try:
        sys.argv = [str(executable), *argv[1:]]
        sys.path[0] = str(executable.parent)
        # Pool tasks resolve top-level functions through their owning module.
        # A detached exec dictionary makes upstream coincidence tasks unpicklable.
        sys.modules["__main__"] = module
        exec(compile(tree, str(executable), "exec"), module.__dict__)
    finally:
        sys.modules["__main__"] = original_main
        sys.argv, sys.path[0] = original_argv, original_path


if __name__ == "__main__":
    main()
