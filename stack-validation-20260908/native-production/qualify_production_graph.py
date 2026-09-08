"""Untimed eager-first, byte-exact verification of every normal graph call."""
import contextlib
import hashlib
import importlib.util
from math import sqrt
import operator
from pathlib import Path
import time

import numpy as np
import torch

from pycbc.filter import _torch_cuda_graph as graphs


spec = importlib.util.spec_from_file_location(
    "original_qualification", Path(__file__).with_name("qualify-inspiral.py"))
donor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(donor)


def snapshot(value):
    tensor = value if type(value) is torch.Tensor else value._data.tensor
    values = tensor.detach().cpu().numpy().copy()
    return str(values.dtype), tuple(values.shape), values.tobytes()


def certificate(value):
    dtype, shape, data = value
    return dict(dtype=dtype, shape=shape, nbytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


class RetainedPair:
    """Predict the one index offset performed by bin/pycbc_inspiral:293."""
    def __init__(self, pair, raw, offset):
        self.pair = pair
        assert tuple(snapshot(value) for value in pair) == raw
        dtype, shape, data = raw[0]
        assert dtype == 'int64' and len(shape) == 1
        offset = operator.index(offset)
        indices = np.frombuffer(data, dtype=np.int64).copy().reshape(shape)
        # Derive expected bytes from the eager-verified return, never from
        # a later snapshot that could already contain graph corruption.
        indices += offset
        self.expected = ((dtype, shape, indices.tobytes()), raw[1])
        self.refs = tuple(self.references(value) for value in pair)
        self.bindings = tuple(self.binding(value) for value in pair)
        versions = [refs[2]._version for refs in self.refs]
        self.expected_versions = [versions[0] + 1, versions[1]]
        self.record = dict(offset=offset, before=[certificate(v) for v in raw],
                           expected_after=[certificate(v) for v in self.expected],
                           bindings=self.bindings, versions_before=versions,
                           expected_versions_after=self.expected_versions,
                           stage='pending', transition_count=0, verified=False)

    @staticmethod
    def references(value):
        data = value if type(value) is torch.Tensor else value._data
        tensor = data if type(data) is torch.Tensor else data.tensor
        return value, data, tensor

    @classmethod
    def binding(cls, value):
        array, data, tensor = cls.references(value)
        return dict(array=id(array), data=id(data), tensor=id(tensor), pointer=tensor.data_ptr(),
                    storage_pointer=tensor.untyped_storage().data_ptr(),
                    storage_bytes=tensor.untyped_storage().nbytes(), shape=list(tensor.shape),
                    stride=list(tensor.stride()), storage_offset=tensor.storage_offset(),
                    dtype=str(tensor.dtype), device=str(tensor.device))

    def verify(self):
        for value, refs in zip(self.pair, self.refs):
            assert all(a is b for a, b in zip(self.references(value), refs))
        assert tuple(self.binding(value) for value in self.pair) == self.bindings
        observed = tuple(snapshot(value) for value in self.pair)
        assert observed == self.expected
        versions = [refs[2]._version for refs in self.refs]
        assert versions == self.expected_versions
        if self.record['stage'] == 'pending':
            self.record.update(stage='complete', transition_count=1, verified=True,
                               observed_after=[certificate(v) for v in observed],
                               observed_versions_after=versions)


class Qualification(donor.Qualification):
    def __init__(self, *args):
        super().__init__(*args)
        self.graphs = dict(captures=[], calls=[], replays=0,
                          retention_checks=0, retained_nonempty=0,
                          capture_contexts=[])
        self.data["offline_graph"] = self.graphs
        self.retained = []
        self.controller = None
        self.active_capture = None

    def verify_retained(self):
        for retained in self.retained:
            retained.verify()
            self.graphs["retention_checks"] += 1

    def install(self):
        super().install()
        from pycbc.filter import matchedfilter

        replay = torch.cuda.CUDAGraph.replay

        def observed_replay(graph):
            result = replay(graph)
            self.graphs["replays"] += 1
            return result

        self.patch(torch.cuda.CUDAGraph, "replay", observed_replay)
        graph_context = torch.cuda.graph

        @contextlib.contextmanager
        def observed_context(*args, **kwargs):
            assert self.active_capture is not None
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.active_capture["setup_and_three_warmups_seconds"] = (
                start - self.active_capture["start_perf"])
            with graph_context(*args, **kwargs):
                yield
            torch.cuda.synchronize()
            self.active_capture["capture_seconds"] = time.perf_counter() - start
            self.graphs["capture_contexts"].append(self.active_capture["segment"])

        self.patch(torch.cuda, "graph", observed_context)
        capture = graphs.capture_symmetric_cuda_graph

        def observed_capture(control, segnum, window, template_norm=1.0):
            if (segnum, window) in getattr(control, "_cuda_graphs", {}):
                return capture(control, segnum, window, template_norm)
            assert self.active_capture is None
            record = dict(segment=segnum, window=window, warmups=3,
                          start_perf=time.perf_counter())
            self.active_capture = record
            try:
                assert capture(control, segnum, window, template_norm) is True
                torch.cuda.synchronize()
                record["total_capture_setup_seconds"] = time.perf_counter() - record.pop("start_perf")
                record["binding"] = repr(graphs._binding(control, segnum, window)[0])
                self.graphs["captures"].append(record)
                return True
            finally:
                self.active_capture = None

        self.patch(graphs, "capture_symmetric_cuda_graph", observed_capture)
        initialize = matchedfilter.MatchedFilterControl.__init__

        def observed_init(control, *args, **kwargs):
            initialize(control, *args, **kwargs)
            assert self.controller is None
            self.controller = control
            signature = tuple(repr(graphs._binding(control, i, 4096)[0])
                              for i in range(len(control.segments)))
            self.graphs["initial_binding"] = signature
            selected = control.matched_filter_and_cluster
            torch.cuda.reset_peak_memory_stats()
            self.graphs["memory_before_calls"] = self.memory()

            def observed_filter(segnum, template_norm, window, epoch=None):
                self.verify_retained()
                assert tuple(repr(graphs._binding(control, i, window)[0]) for i in range(len(control.segments))) == signature
                assert all(entry.matches(graphs._binding(control, segnum, window), control.threshold_and_clusterers[segnum], window) for (segnum, window), entry in control._cuda_graphs.items()) if hasattr(control, "_cuda_graphs") else True
                inputs = (snapshot(control.htilde), snapshot(control.segments[segnum]))
                norm = (4.0 * control.delta_f) / sqrt(template_norm)
                control.correlators[segnum].correlate()
                control.ifft.execute()
                eager_values, eager_indices = control.threshold_and_clusterers[
                    segnum].threshold_and_cluster(control.snr_threshold / norm, window)
                expected_full = (snapshot(control.corr_mem), snapshot(control.snr_mem))
                expected_sparse = (snapshot(eager_indices), snapshot(eager_values))
                self.verify_retained()
                first_use = (segnum, window) not in getattr(control, "_cuda_graphs", {})
                before = self.graphs["replays"]
                start = time.perf_counter() if first_use else None
                result = selected(segnum, template_norm, window, epoch=epoch)
                assert self.graphs["replays"] == before + 1
                actual_full = (snapshot(control.corr_mem), snapshot(control.snr_mem))
                assert actual_full == expected_full, "Full correlation/SNR byte mismatch"
                assert (snapshot(control.htilde), snapshot(control.segments[segnum])) == inputs
                assert tuple(repr(graphs._binding(control, i, window)[0]) for i in range(len(control.segments))) == signature
                assert all(entry.matches(graphs._binding(control, segnum, window), control.threshold_and_clusterers[segnum], window) for (segnum, window), entry in control._cuda_graphs.items()) if hasattr(control, "_cuda_graphs") else True
                retained = None
                if len(eager_indices) == 0:
                    assert result == ([], [], [], [], []) or result == [[], [], [], [], []]
                    sparse = expected_sparse
                else:
                    assert len(result) == 5 and result[1] == norm
                    assert result[0]._data.tensor.data_ptr() == control.snr_mem._data.tensor.data_ptr()
                    assert result[2]._data.tensor.data_ptr() == control.corr_mem._data.tensor.data_ptr()
                    sparse = (snapshot(result[3]), snapshot(result[4]))
                    assert sparse == expected_sparse, "Sparse result byte mismatch"
                    scratch = control.threshold_and_clusterers[segnum]
                    forbidden = {v.untyped_storage().data_ptr() for v in (
                        control.snr_mem._data.tensor, control.corr_mem._data.tensor,
                        scratch._triton_block_max, scratch._triton_block_idx, scratch._triton_keep)}
                    assert all(v._data.tensor.untyped_storage().data_ptr() not in forbidden
                               for v in result[3:])
                    self.verify_retained()
                    segment = control.segments[segnum]
                    assert operator.index(segment.cumulative_index) == (
                        segment.seg_slice.start + segment.analyze.start)
                    retained = RetainedPair(result[3:], sparse,
                                            segment.cumulative_index)
                    self.graphs["retained_nonempty"] += 1
                self.verify_retained()
                row = dict(call=len(self.graphs["calls"]), template=list(self.current_template),
                           segment=segnum, template_norm=float(template_norm), norm=float(norm),
                           raw_threshold=float(control.snr_threshold / norm),
                           inputs=[certificate(v) for v in inputs],
                           full=[certificate(v) for v in actual_full],
                           sparse=[certificate(v) for v in sparse],
                           full_bytes_equal=True, sparse_bytes_equal=True,
                           inputs_unchanged=True, binding_unchanged=True,
                           empty=len(eager_indices) == 0)
                if first_use:
                    row["first_use_seconds_with_verification"] = time.perf_counter() - start
                if retained is not None:
                    row['consumer'] = retained.record
                    # Activate the post-consumer expectation only at return.
                    self.retained = [self.retained[0], retained] if self.retained else [retained]
                self.graphs["calls"].append(row)
                return result

            self.patch(control, "matched_filter_and_cluster", observed_filter)

        self.patch(matchedfilter.MatchedFilterControl, "__init__", observed_init)

    @staticmethod
    def memory():
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved())

    def checks(self):
        checks = super().checks()
        checks.pop("scalar_ifft_executed_for_every_pair")
        primary_id = self.data["matched_filter_controllers"][0]["ifft_engine_id"]
        primary = self.data["fft_engines"][primary_id]
        checks["python_ifft_calls_1920_oracles_plus_20_warmup_capture"] = (
            primary["nbatch"] == 1 and primary["execute_attempts"] == primary["execute_successes"] == 1940)
        checks["five_real_captures"] = [v["segment"] for v in self.graphs["captures"]] == list(range(5))
        checks["five_capture_contexts"] = self.graphs["capture_contexts"] == list(range(5))
        checks["exactly_1920_actual_replays_and_full_comparisons"] = (
            self.graphs["replays"] == len(self.graphs["calls"]) == 1920)
        checks["templates_and_norms_changed"] = (
            len({v["inputs"][0]["sha256"] for v in self.graphs["calls"]}) == 384
            and len({v["template_norm"] for v in self.graphs["calls"]}) > 1)
        self.verify_retained()
        checks["retained_sparse_outputs_survived"] = (
            self.graphs["retained_nonempty"] > 1 and self.graphs["retention_checks"] >= 1920)
        checks['consumer_offsets_verified_for_every_nonempty_return'] = all(
            row['empty'] or row['consumer']['verified'] for row in self.graphs['calls'])
        if self.controller is not None:
            assert tuple(repr(graphs._binding(self.controller, i, 4096)[0]) for i in range(len(self.controller.segments))) == self.graphs["initial_binding"]
            self.graphs["final_bindings"] = [dict(segment=k[0], window=k[1], signature=repr(v.signature))
                for k, v in self.controller._cuda_graphs.items()]
            self.graphs["memory_after_calls"] = self.memory()
            self.controller.clear_cuda_graphs()
            checks["explicit_graph_cleanup"] = not self.controller._cuda_graphs
        return checks


if __name__ == "__main__":
    donor.Qualification = Qualification
    raise SystemExit(donor.main())
