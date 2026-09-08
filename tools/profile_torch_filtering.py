#!/usr/bin/env python3
"""Trace an unchanged pycbc_inspiral CUDA template loop in a separate process.

Run only under the shared measurement lock after preceding campaigns exit.
Semantic ranges do not synchronize; only whole-loop boundaries synchronize.
This instrumented run is ineligible for throughput. Validate its HDF output
against a separate unprofiled same-source run using unchanged science budgets.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Executable argv is required after --')
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from pycbc.events.eventmgr import EventManager
    from pycbc.events.threshold_torch import TorchThresholdCluster
    from pycbc.fft.torchfft import IFFT
    from pycbc.filter.matchedfilter_torch import TorchCorrelator
    from pycbc.scheme import mgr
    from pycbc.types import array_torch
    from pycbc.vetoes import chisq_torch
    from pycbc.waveform.bank import FilterBank

    profiler = None
    active = False
    loop_range = None
    device = None
    patches = []
    record = dict(state='running', command=command, pid=os.getpid(),
                  cwd=os.getcwd(), torch_version=torch.__version__,
                  torch_cuda_version=torch.version.cuda,
                  affinity=sorted(os.sched_getaffinity(0)), indices=[],
                  start_monotonic_ns=None, stop_monotonic_ns=None,
                  instrumented=True, eligible_for_throughput=False,
                  interval='First bank lookup through event consolidation',
                  semantic_calls={}, device_name=None,
                  helper_sha256=hashlib.sha256(
                      Path(__file__).read_bytes()).hexdigest())

    def save():
        path = args.output / 'receipt.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(path)

    def thread_counts():
        return dict(intra_op=torch.get_num_threads(),
                    inter_op=torch.get_num_interop_threads())

    def require_single_thread():
        if thread_counts() != dict(intra_op=1, inter_op=1):
            raise ValueError('Both Torch thread counts must be one')

    def patch(owner, name, function):
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, function)

    def annotate(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*values, **kwargs):
            if not active:
                return original(*values, **kwargs)
            record['semantic_calls'][label] = record['semantic_calls'].get(
                label, 0) + 1
            with torch.profiler.record_function('pycbc::' + label):
                return original(*values, **kwargs)

        patch(owner, name, wrapped)

    getitem = FilterBank.__getitem__
    performance = EventManager.save_performance

    def observed_getitem(bank, index):
        nonlocal profiler, active, device, loop_range
        if profiler is None:
            record['threads_at_first_bank'] = thread_counts()
            require_single_thread()
            device = getattr(mgr.state, 'device', None)
            if getattr(device, 'type', None) != 'cuda':
                raise ValueError('This helper requires the CUDA scheme')
            record['device_name'] = torch.cuda.get_device_name(device)
            record['bank_templates'] = len(bank)
            torch.cuda.synchronize(device)
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False)
            profiler.start()
            active = True
            loop_range = torch.profiler.record_function('pycbc::filter_loop')
            loop_range.__enter__()
            record['start_monotonic_ns'] = time.monotonic_ns()
        record['indices'].append(int(index))
        with torch.profiler.record_function('pycbc::bank_lookup'):
            return getitem(bank, index)

    def observed_performance(manager, ncores, nfilters,
                             ntemplates, *rest, **kwargs):
        nonlocal active, loop_range
        if not active:
            raise ValueError('Performance metadata arrived outside the loop')
        with torch.profiler.record_function('pycbc::boundary_sync'):
            torch.cuda.synchronize(device)
        record['stop_monotonic_ns'] = time.monotonic_ns()
        loop_range.__exit__(None, None, None)
        loop_range = None
        active = False
        profiler.stop()
        record['threads_at_loop_end'] = thread_counts()
        require_single_thread()
        record.update(completed_templates=ntemplates,
                      segments_per_template=nfilters,
                      host_cores=ncores)
        return performance(manager, ncores, nfilters,
                           ntemplates, *rest, **kwargs)

    previous_argv = sys.argv
    try:
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        record['threads_before_executable'] = thread_counts()
        require_single_thread()
        patch(FilterBank, '__getitem__', observed_getitem)
        patch(EventManager, 'save_performance', observed_performance)
        annotate(IFFT, 'execute', 'ifft')
        annotate(TorchCorrelator, 'correlate', 'correlation')
        annotate(TorchThresholdCluster, 'threshold_and_cluster', 'threshold')
        annotate(
            chisq_torch,
            'power_chisq_at_points_from_precomputed',
            'chisq')
        annotate(array_torch, 'squared_norm', 'squared_norm')
        annotate(array_torch, 'inner', 'normalization')
        annotate(array_torch, 'numpy', 'to_numpy')
        save()
        sys.argv = command
        try:
            runpy.run_path(command[0], run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        if (active or profiler is None or
                record['indices'] != list(range(record['bank_templates'])) or
                record['completed_templates'] != record['bank_templates']):
            raise ValueError('Incomplete template-loop coverage')
        expected = (record['completed_templates'] *
                    record['segments_per_template'])
        if record['semantic_calls'].get('ifft') != expected:
            raise ValueError(
                'IFFT coverage differs from template/segment pairs')
        profiler.export_chrome_trace(str(args.output / 'trace.json'))
        record['filtering_window_seconds'] = (
            record['stop_monotonic_ns'] - record['start_monotonic_ns']) / 1e9
        record['trace_sha256'] = hashlib.sha256(
            (args.output / 'trace.json').read_bytes()).hexdigest()
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed', error=repr(exc))
        raise
    finally:
        if loop_range is not None:
            loop_range.__exit__(None, None, None)
        if active:
            profiler.stop()
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
        sys.argv = previous_argv
        save()


if __name__ == '__main__':
    main()
