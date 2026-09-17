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
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time


def hash_file_chunks(path, chunk_size=65536):
    """Compute SHA256 of path using 64 KiB chunks."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


TARGET_MODULE_NAMES = (
    'pycbc.filter.gpu_search.core',
    'pycbc.filter.gpu_search.adapter',
    'pycbc.filter.gpu_search.engine',
    'pycbc.filter.gpu_search.graphs',
    'pycbc.filter.gpu_search.candidates',
    'pycbc.filter.gpu_search.vetoes',
    'pycbc.filter.matchedfilter_torch',
    'pycbc.fft.torchfft',
    'pycbc.events.threshold_torch',
    'pycbc.vetoes.chisq_torch',
    'pycbc.types.array_torch',
)


def collect_source_identity(helper_file):
    """Hash loaded target modules and helper script."""
    sources = {}
    helper_path = Path(helper_file).resolve()
    if helper_path.is_file():
        sources['tools.profile_torch_filtering'] = {
            'path': str(helper_path),
            'sha256': hash_file_chunks(helper_path),
        }
    for mod_name in TARGET_MODULE_NAMES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        mod_file = getattr(mod, '__file__', None)
        if mod_file is not None:
            try:
                p = Path(mod_file).resolve()
                if p.is_file():
                    sources[mod_name] = {
                        'path': str(p),
                        'sha256': hash_file_chunks(p),
                    }
            except (OSError, ValueError):
                pass
    return sources


def collect_input_identity(argv):
    """Record metadata for existing files in command arguments."""
    inputs = {}
    for arg in argv:
        if isinstance(arg, (str, Path, os.PathLike)):
            try:
                if os.path.isfile(arg):
                    p = Path(arg).resolve()
                    stat = p.stat()
                    inputs[str(arg)] = {
                        'path': str(p),
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'sha256': hash_file_chunks(p),
                    }
            except (OSError, ValueError):
                pass
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--cpu', action='store_true', default=False,
        help='Opt-in to Torch CPU tracing without CUDA requirements'
    )
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

    search_core = sys.modules.get('pycbc.filter.gpu_search.core')
    if search_core is None:
        try:
            from pycbc.filter.gpu_search import core as search_core
        except (ImportError, AttributeError):
            search_core = None

    search_adapter = sys.modules.get('pycbc.filter.gpu_search.adapter')
    if search_adapter is None:
        try:
            from pycbc.filter.gpu_search import adapter as search_adapter
        except (ImportError, AttributeError):
            search_adapter = None

    search_candidates = sys.modules.get(
        'pycbc.filter.gpu_search.candidates'
    )
    if search_candidates is None:
        try:
            from pycbc.filter.gpu_search import (
                candidates as search_candidates,
            )
        except (ImportError, AttributeError):
            search_candidates = None

    search_vetoes = sys.modules.get('pycbc.filter.gpu_search.vetoes')
    if search_vetoes is None:
        try:
            from pycbc.filter.gpu_search import vetoes as search_vetoes
        except (ImportError, AttributeError):
            search_vetoes = None

    profiler = None
    active = False
    loop_range = None
    device = None
    patches = []
    allow_cpu = bool(args.cpu)

    exe_path = Path(command[0])
    exe_resolved = (
        str(exe_path.resolve()) if exe_path.exists() else command[0]
    )
    exe_sha256 = (
        hash_file_chunks(Path(exe_resolved))
        if Path(exe_resolved).is_file()
        else None
    )
    helper_path = Path(__file__).resolve()
    helper_sha256 = (
        hash_file_chunks(helper_path)
        if helper_path.is_file()
        else None
    )
    source_identity = collect_source_identity(__file__)
    input_identity = collect_input_identity(command)

    input_source_identity = {
        'command': command,
        'executable': exe_resolved,
        'executable_sha256': exe_sha256,
        'helper_sha256': helper_sha256,
        'sources': source_identity,
        'inputs': input_identity,
    }

    record = dict(
        state='running',
        command=command,
        pid=os.getpid(),
        cwd=os.getcwd(),
        torch_version=torch.__version__,
        torch_cuda_version=getattr(torch.version, 'cuda', None),
        affinity=(
            sorted(os.sched_getaffinity(0))
            if hasattr(os, 'sched_getaffinity')
            else None
        ),
        indices=[],
        start_monotonic_ns=None,
        stop_monotonic_ns=None,
        instrumented=True,
        eligible_for_throughput=False,
        interval='First bank lookup through event consolidation',
        semantic_calls={},
        tile_calls=0,
        executed_template_rows=0,
        tile_sizes=[],
        route=None,
        device=None,
        device_name=None,
        source_identity=source_identity,
        input_identity=input_identity,
        input_source_identity=input_source_identity,
        helper_sha256=helper_sha256,
    )

    def save():
        path = args.output / 'receipt.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(path)

    def thread_counts():
        return dict(
            intra_op=torch.get_num_threads(),
            inter_op=torch.get_num_interop_threads(),
        )

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
            record['semantic_calls'][label] = (
                record['semantic_calls'].get(label, 0) + 1
            )
            with torch.profiler.record_function('pycbc::' + label):
                return original(*values, **kwargs)

        patch(owner, name, wrapped)

    _in_core_correlate = False

    def make_correlate_wrapper(orig_func):
        @functools.wraps(orig_func)
        def observed_correlate_and_ifft(
            templates, data, cout_workspace, out_workspace, tlen,
            *extra_args, **extra_kwargs
        ):
            nonlocal _in_core_correlate
            if _in_core_correlate or not active:
                return orig_func(
                    templates, data, cout_workspace, out_workspace, tlen,
                    *extra_args, **extra_kwargs
                )
            _in_core_correlate = True
            try:
                b = extra_kwargs.get('batch_size')
                if b is None and len(extra_args) >= 4:
                    b = extra_args[3]
                if b is None:
                    try:
                        sig = inspect.signature(orig_func)
                        bound = sig.bind(
                            templates, data, cout_workspace, out_workspace,
                            tlen, *extra_args, **extra_kwargs
                        )
                        b = bound.arguments.get('batch_size')
                    except (TypeError, ValueError):
                        pass
                if b is None:
                    ndim = getattr(templates, 'ndim', None)
                    shape = getattr(templates, 'shape', None)
                    if ndim == 1 or (shape is not None and len(shape) == 1):
                        b = 1
                    elif ndim == 2 or (shape is not None and len(shape) == 2):
                        b = shape[0]
                    elif isinstance(templates, (list, tuple)):
                        b = len(templates)
                    else:
                        b = 1
                b = int(b)
                record['tile_calls'] += 1
                record['executed_template_rows'] += b
                record['tile_sizes'].append(b)
                record['semantic_calls']['core_correlate_and_ifft'] = (
                    record['semantic_calls'].get(
                        'core_correlate_and_ifft', 0
                    ) + 1
                )
                with torch.profiler.record_function(
                    'pycbc::core_correlate_and_ifft'
                ):
                    return orig_func(
                        templates, data, cout_workspace, out_workspace, tlen,
                        *extra_args, **extra_kwargs
                    )
            finally:
                _in_core_correlate = False

        return observed_correlate_and_ifft

    _in_select_candidates = False

    def make_candidates_wrapper(orig_func):
        @functools.wraps(orig_func)
        def observed_select_candidates(*c_args, **c_kwargs):
            nonlocal _in_select_candidates
            if _in_select_candidates or not active:
                return orig_func(*c_args, **c_kwargs)
            _in_select_candidates = True
            try:
                record['semantic_calls']['select_candidates'] = (
                    record['semantic_calls'].get(
                        'select_candidates', 0
                    ) + 1
                )
                with torch.profiler.record_function(
                    'pycbc::select_candidates'
                ):
                    return orig_func(*c_args, **c_kwargs)
            finally:
                _in_select_candidates = False

        return observed_select_candidates

    _in_batched_chisq = False

    def make_chisq_wrapper(orig_func):
        @functools.wraps(orig_func)
        def observed_batched_chisq(*v_args, **v_kwargs):
            nonlocal _in_batched_chisq
            if _in_batched_chisq or not active:
                return orig_func(*v_args, **v_kwargs)
            _in_batched_chisq = True
            try:
                record['semantic_calls']['batched_chisq'] = (
                    record['semantic_calls'].get('batched_chisq', 0) + 1
                )
                with torch.profiler.record_function(
                    'pycbc::batched_power_chisq'
                ):
                    return orig_func(*v_args, **v_kwargs)
            finally:
                _in_batched_chisq = False

        return observed_batched_chisq

    getitem = FilterBank.__getitem__
    performance = EventManager.save_performance

    def observed_getitem(bank, index):
        nonlocal profiler, active, device, loop_range
        if profiler is None:
            record['threads_at_first_bank'] = thread_counts()
            require_single_thread()
            device = getattr(mgr.state, 'device', None)
            dev_type = getattr(device, 'type', None)
            if not allow_cpu and dev_type != 'cuda':
                raise ValueError(
                    'This helper requires the CUDA scheme '
                    '(use --cpu for Torch CPU)'
                )
            if allow_cpu and dev_type not in ('cpu', 'cuda'):
                raise ValueError(
                    f'Unsupported device {device} under --cpu'
                )

            record['device'] = str(device)
            if dev_type == 'cuda':
                record['device_name'] = torch.cuda.get_device_name(device)
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                torch.cuda.synchronize(device)
            else:
                record['device_name'] = 'cpu'
                activities = [torch.profiler.ProfilerActivity.CPU]

            record['bank_templates'] = len(bank)
            bank_file = getattr(bank, 'filename', None)
            if bank_file is not None:
                record['input_source_identity']['bank_file'] = str(bank_file)

            profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
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
        dev_type = getattr(device, 'type', None)
        if dev_type == 'cuda':
            with torch.profiler.record_function('pycbc::boundary_sync'):
                torch.cuda.synchronize(device)
        record['stop_monotonic_ns'] = time.monotonic_ns()
        loop_range.__exit__(None, None, None)
        loop_range = None
        active = False
        profiler.stop()
        record['threads_at_loop_end'] = thread_counts()
        require_single_thread()
        record.update(
            completed_templates=ntemplates,
            segments_per_template=nfilters,
            host_cores=ncores,
        )
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
            'chisq',
        )
        annotate(array_torch, 'squared_norm', 'squared_norm')
        annotate(array_torch, 'inner', 'normalization')
        annotate(array_torch, 'numpy', 'to_numpy')

        # Tiled shared-core instrumentation
        if search_core and hasattr(search_core, 'correlate_and_ifft'):
            patch(
                search_core,
                'correlate_and_ifft',
                make_correlate_wrapper(
                    getattr(search_core, 'correlate_and_ifft')
                ),
            )
        if search_adapter and hasattr(search_adapter, 'correlate_and_ifft'):
            patch(
                search_adapter,
                'correlate_and_ifft',
                make_correlate_wrapper(
                    getattr(search_adapter, 'correlate_and_ifft')
                ),
            )
        if search_candidates and hasattr(
            search_candidates, 'select_tile_candidates'
        ):
            patch(
                search_candidates,
                'select_tile_candidates',
                make_candidates_wrapper(
                    getattr(search_candidates, 'select_tile_candidates')
                ),
            )
        if (
            search_adapter
            and hasattr(search_adapter, 'select_tile_candidates')
        ):
            patch(
                search_adapter,
                'select_tile_candidates',
                make_candidates_wrapper(
                    getattr(search_adapter, 'select_tile_candidates')
                ),
            )
        if search_vetoes and hasattr(search_vetoes, 'batched_power_chisq'):
            patch(
                search_vetoes,
                'batched_power_chisq',
                make_chisq_wrapper(
                    getattr(search_vetoes, 'batched_power_chisq')
                ),
            )
        if search_adapter and hasattr(search_adapter, 'batched_power_chisq'):
            patch(
                search_adapter,
                'batched_power_chisq',
                make_chisq_wrapper(
                    getattr(search_adapter, 'batched_power_chisq')
                ),
            )

        if hasattr(torch, 'mul'):
            orig_torch_mul = torch.mul

            @functools.wraps(orig_torch_mul)
            def observed_core_mul(*m_args, **m_kwargs):
                if not (_in_core_correlate and active):
                    return orig_torch_mul(*m_args, **m_kwargs)
                record['semantic_calls']['core_mul'] = (
                    record['semantic_calls'].get('core_mul', 0) + 1
                )
                with torch.profiler.record_function('pycbc::core_mul'):
                    return orig_torch_mul(*m_args, **m_kwargs)

            patch(torch, 'mul', observed_core_mul)

        fft_mod = getattr(torch, 'fft', None)
        if fft_mod is not None and hasattr(fft_mod, 'ifft'):
            orig_torch_ifft = fft_mod.ifft

            @functools.wraps(orig_torch_ifft)
            def observed_core_ifft(*i_args, **i_kwargs):
                if not (_in_core_correlate and active):
                    return orig_torch_ifft(*i_args, **i_kwargs)
                record['semantic_calls']['core_ifft'] = (
                    record['semantic_calls'].get('core_ifft', 0) + 1
                )
                with torch.profiler.record_function('pycbc::core_ifft'):
                    return orig_torch_ifft(*i_args, **i_kwargs)

            patch(fft_mod, 'ifft', observed_core_ifft)

        save()
        sys.argv = command
        try:
            runpy.run_path(command[0], run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise

        record['source_identity'].update(
            collect_source_identity(__file__)
        )
        record['input_source_identity']['sources'] = (
            record['source_identity']
        )

        if (
            active
            or profiler is None
            or record['indices'] != list(range(record['bank_templates']))
            or record['completed_templates'] != record['bank_templates']
        ):
            raise ValueError('Incomplete template-loop coverage')

        expected = (
            record['completed_templates'] * record['segments_per_template']
        )
        tiled_rows = record.get('executed_template_rows', 0)
        scalar_iffts = record['semantic_calls'].get('ifft', 0)

        if tiled_rows > 0 and scalar_iffts == 0:
            if tiled_rows != expected:
                raise ValueError(
                    f'Tiled template row coverage ({tiled_rows}) differs '
                    f'from template/segment pairs ({expected})'
                )
            record['route'] = 'tiled'
        elif scalar_iffts > 0 and tiled_rows == 0:
            if scalar_iffts != expected:
                raise ValueError(
                    f'IFFT coverage ({scalar_iffts}) differs from '
                    f'template/segment pairs ({expected})'
                )
            record['route'] = 'scalar'
        elif tiled_rows > 0 and scalar_iffts > 0:
            if tiled_rows + scalar_iffts != expected:
                raise ValueError(
                    f'Combined matched-filter coverage '
                    f'({tiled_rows + scalar_iffts}) differs from '
                    f'template/segment pairs ({expected})'
                )
            record['route'] = 'mixed'
        else:
            raise ValueError(
                f'No filtering operations recorded; expected {expected} '
                f'template/segment pairs'
            )

        profiler.export_chrome_trace(str(args.output / 'trace.json'))
        record['filtering_window_seconds'] = (
            record['stop_monotonic_ns'] - record['start_monotonic_ns']
        ) / 1e9
        record['trace_sha256'] = hash_file_chunks(
            args.output / 'trace.json'
        )
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
