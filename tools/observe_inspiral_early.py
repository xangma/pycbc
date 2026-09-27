#!/usr/bin/env python
"""Opt-in, bounded early-selection trace for a 32-template inspiral run.

Run this wrapper around either the pristine or candidate executable, with
``PYCBC_EARLY_TRACE`` naming a new HDF file and
``PYCBC_EARLY_EXPECTED_SEGMENTS`` set to the predeclared segment count.
``PYCBC_EARLY_EXPECTED_TEMPLATES`` defaults to 32; setting it to 1 supports
a small CPU smoke run. The comparator accepts only complete 32-template traces.
The executable is compiled with in-memory observation calls; its calculations
and source files are unchanged. Traces from the two arms are compared with
``benchmark_early_decisions.compare_early_decisions``.
"""

import ast
import os
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from tools.observe_pycbc_inspiral import _array
    from tools.benchmark_early_decisions import summarize_series, replay_summary
except ImportError:
    from observe_pycbc_inspiral import _array
    from benchmark_early_decisions import summarize_series, replay_summary


def _call_on(node, method):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "matched_filter"
    )


def _observed_tree(source, filename):
    tree = ast.parse(source, filename=filename)
    if any(isinstance(node, ast.Name) and node.id == "__pycbc_early" for node in ast.walk(tree)):
        raise RuntimeError("early observer namespace already used")
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    scalar_count = sum(_call_on(node.value, "matched_filter_and_cluster") for node in assignments)
    batch_count = sum(_call_on(node.value, "batched_matched_filter_and_cluster")
                      and any(isinstance(target, ast.Name) and target.id == "batch_results"
                              for target in node.targets) for node in assignments)
    if scalar_count != 1 or batch_count not in (0, 1):
        raise RuntimeError("unsupported matched-filter call boundaries")
    final_writes = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "write_events"
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "event_mgr"]
    if len(final_writes) not in (1, 2):
        raise RuntimeError("unsupported event output boundaries")
    final_write = max(final_writes, key=lambda node: node.lineno)

    def expr(method, names):
        return ast.Expr(ast.Call(
            ast.Attribute(ast.Name("__pycbc_early", ast.Load()), method, ast.Load()),
            [ast.Name(name, ast.Load()) for name in names], [],
        ))

    class Inject(ast.NodeTransformer):
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if _call_on(node.value, "matched_filter_and_cluster"):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
                    raise RuntimeError("unsupported scalar matched-filter assignment")
                names = [item.id for item in node.targets[0].elts if isinstance(item, ast.Name)]
                if names != ["snr", "norm", "corr", "idx", "snrv"]:
                    raise RuntimeError("unsupported scalar matched-filter result")
                after = expr("scalar", [
                    "t_num", "s_num", "stilde", "template", "matched_filter",
                    "sigmasq", "cluster_window", "norm", "idx", "snrv",
                ])
                return [node, ast.copy_location(after, node)]
            if _call_on(node.value, "batched_matched_filter_and_cluster"):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != "batch_results":
                    return node  # Setup warmup is outside the search decision.
                before = expr("begin_batch", [])
                after = expr("batch", [
                    "batch_tnums", "active_indices", "s_num", "stilde",
                    "active_templates", "matched_filter", "cluster_window", "batch_results",
                ])
                return [ast.copy_location(before, node), node, ast.copy_location(after, node)]
            return node

        def visit_Expr(self, node):
            node = self.generic_visit(node)
            if node is final_write:
                return [node, ast.copy_location(expr("finish", []), node)]
            return node

    tree = Inject().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


class EarlyObserver:
    def __init__(self, path, expected_segments, expected_templates=32):
        if expected_segments <= 0 or expected_segments > 64:
            raise ValueError("expected segment count must be in 1..64")
        if expected_templates <= 0 or expected_templates > 32:
            raise ValueError("expected template count must be in 1..32")
        self.output = h5py.File(path, "x")
        self.output.attrs["schema"] = "pycbc-early-selection-v1"
        self.output.attrs["complete"] = False
        self.output.attrs["expected_templates"] = expected_templates
        self.output.attrs["expected_segments"] = expected_segments
        self.rows = self.output.create_group("rows")
        self.expected_segments = expected_segments
        self.expected_templates = expected_templates
        self.pending = None
        self._patched = False

    def _save(self, template_index, segment_index, template, stilde,
              mf_control, window, norm, idx, snrv, valid_snr, backend):
        t, s = int(template_index), int(segment_index)
        if not 0 <= t < self.expected_templates or not 0 <= s < self.expected_segments:
            raise RuntimeError("observation is outside predeclared template/segment grid")
        name = f"{t:03d}-{s:03d}"
        if name in self.rows:
            raise RuntimeError("duplicate early observation")
        series = np.asarray(_array(valid_snr), dtype=np.complex64)
        selected = np.asarray(_array(idx), dtype=np.int64)
        values = np.asarray(_array(snrv), dtype=np.complex64)
        if series.ndim != 1 or not len(series) or len(selected) != len(values):
            raise RuntimeError("malformed early selection arrays")
        if not np.isfinite(series).all() or not np.isfinite(values).all():
            raise RuntimeError("nonfinite early selection arrays")
        if int(stilde.analyze.stop - stilde.analyze.start) != len(series):
            raise RuntimeError("valid SNR series does not cover analysis interval")
        if np.any(selected < 0) or np.any(selected >= len(series)):
            raise RuntimeError("selected index is outside valid SNR series")
        if not np.array_equal(series[selected], values):
            raise RuntimeError("selected SNR values disagree with captured series")
        norm_value = float(norm)
        if not np.isfinite(norm_value) or norm_value <= 0:
            raise RuntimeError("invalid filter normalization")
        threshold = float(mf_control.snr_threshold)
        if backend == "cpu":
            raw_threshold = np.float32(threshold / norm_value)
            threshold_sq = float(np.float32(raw_threshold * raw_threshold))
        elif backend == "jax":
            raw_threshold = threshold / norm
            threshold_sq = float(np.asarray(raw_threshold * raw_threshold))
        else:
            raise RuntimeError("unsupported early selection backend")
        top_index, top_score, runner_index, runner_score = summarize_series(
            series, int(window)
        )
        expected = replay_summary(top_index, top_score.copy(), threshold_sq)
        if not np.array_equal(selected, expected):
            raise RuntimeError("native cluster selection disagrees with window summary")
        row = self.rows.create_group(name)
        row.attrs["template_index"] = t
        row.attrs["segment_index"] = s
        row.attrs["template_hash"] = int(template.params.template_hash)
        row.attrs["window"] = int(window)
        row.attrs["threshold_sq"] = threshold_sq
        row.attrs["series_length"] = len(series)
        row.attrs["backend"] = backend
        row.attrs["normalization"] = norm_value
        row.attrs["analyze_start"] = int(stilde.analyze.start)
        row.create_dataset("top_index", data=top_index)
        row.create_dataset("top_score_sq", data=top_score)
        row.create_dataset("runner_index", data=runner_index)
        row.create_dataset("runner_score_sq", data=runner_score)
        row.create_dataset("selected_index", data=selected)

    def scalar(self, template_index, segment_index, stilde, template,
               mf_control, sigmasq, window, norm, idx, snrv):
        if mf_control.cluster_function != "symmetric" or not hasattr(
            mf_control, "threshold_and_clusterers"
        ) or mf_control.threshold_and_clusterers[segment_index].__class__.__module__ != "pycbc.events.threshold_cpu":
            raise RuntimeError("scalar early observer supports CPU symmetric clustering only")
        if not len(idx):
            norm = (4.0 * mf_control.delta_f) / np.sqrt(sigmasq)
        self._save(template_index, segment_index, template, stilde, mf_control,
                   window, norm, idx, snrv, mf_control.snr_mem[stilde.analyze], "cpu")

    def begin_batch(self):
        if self.pending is not None:
            raise RuntimeError("nested or unfinished JAX batch observation")
        if not self._patched:
            from pycbc.filter import matchedfilter_jax as module
            for method, retains_snr in (
                ("_batched_filter_and_cluster", True),
                ("_batched_filter_and_cluster_lean", False),
            ):
                original = getattr(module, method)

                def observed(
                    *args, _original=original, _retains_snr=retains_snr,
                    **kwargs
                ):
                    result = _original(*args, **kwargs)
                    if self.pending is not None:
                        if self.pending is not False:
                            raise RuntimeError("multiple filter kernels in one batch")
                        valid_start, valid_stop = int(args[6]), int(args[7])
                        if _retains_snr:
                            snr_series = result[0]
                        else:
                            # Diagnostic-only reconstruction keeps the
                            # production lean kernel free of a full SNR output.
                            import jax.numpy as jnp
                            kmin, kmax, tlen = map(int, args[3:6])
                            corr_slice = result[0]
                            qtilde = jnp.pad(
                                corr_slice,
                                ((0, 0), (kmin, tlen - kmax)),
                            )
                            snr_series = jnp.fft.ifft(qtilde, axis=-1) * tlen
                        self.pending = snr_series[:, valid_start:valid_stop]
                    return result

                setattr(module, method, observed)
            self._patched = True
        self.pending = False

    def batch(self, batch_tnums, active_indices, segment_index, stilde,
              templates, mf_control, window, results):
        if mf_control.cluster_function != "symmetric" or self.pending is False or self.pending is None:
            raise RuntimeError("JAX batched filter series was not observed")
        series = np.asarray(self.pending, dtype=np.complex64)
        self.pending = None
        if series.ndim != 2 or len(series) != len(active_indices) or len(results) != len(series):
            raise RuntimeError("batched filter/result dimensions disagree")
        for row_index, active_index in enumerate(active_indices):
            _, norm, _, idx, snrv = results[row_index]
            self._save(batch_tnums[active_index], segment_index,
                       templates[row_index], stilde, mf_control, window,
                       norm, idx, snrv, series[row_index], "jax")

    def finish(self):
        if self.pending is not None:
            raise RuntimeError("unfinished JAX batch observation")
        if len(self.rows) != self.expected_templates * self.expected_segments:
            raise RuntimeError("incomplete template/segment early observation grid")
        self.output.attrs["complete"] = True
        self.output.flush()
        self.output.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: PYCBC_EARLY_TRACE=NEW.hdf PYCBC_EARLY_EXPECTED_SEGMENTS=N "
              "observe_inspiral_early.py PYCBC_INSPIRAL [ARGS...]")
        return 0 if argv else 2
    path = os.environ.get("PYCBC_EARLY_TRACE")
    segments = os.environ.get("PYCBC_EARLY_EXPECTED_SEGMENTS")
    if not path or not segments:
        raise RuntimeError("early trace path and predeclared segment count are required")
    script = Path(argv[0]).resolve()
    tree = _observed_tree(script.read_text(), str(script))
    sys.argv = [str(script)] + argv[1:]
    sys.path[0] = str(script.parent)
    namespace = {
        "__name__": "__main__", "__file__": str(script), "__package__": None,
        "__cached__": None, "__pycbc_early": EarlyObserver(
            path, int(segments), int(os.environ.get("PYCBC_EARLY_EXPECTED_TEMPLATES", "32"))
        ),
    }
    exec(compile(tree, str(script), "exec"), namespace, namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
