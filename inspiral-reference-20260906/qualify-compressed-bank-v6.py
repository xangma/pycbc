#!/usr/bin/env python3
"""Require exact Torch CPU/CPU and Torch CUDA/v5 parity on all FFT grids."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_info(source):
    return {name: subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
            for name, args in [('commit', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain'])]}


def main():
    root = Path(__file__).resolve().parent
    source = root / 'source-v6'
    output = root / 'compressed-bank-v6.json'
    require(not output.exists(), 'Qualification output already exists')
    receipt = json.loads((root / 'source-v6.json').read_text())
    before = source_info(source)
    require(before == dict(commit=receipt['commit'], status=''), 'Unexpected source')
    cfg = json.loads((root / 'config.json').read_text())
    for key, value in cfg['environment'].items():
        require(os.environ.get(key) == value, f'Wrong environment: {key}')
    require(os.sched_getaffinity(0) == {cfg['core']}, 'Wrong CPU affinity')
    bank = root / 'inputs/bank-compressed-1e5.hdf'
    compression = json.loads((root / 'compression-1e5.json').read_text())
    require(compression['output_sha256'][str(bank)] == digest(bank), 'Compressed bank changed')
    paths = [Path(__file__), root / 'source-v6.json', root / 'config.json',
             root / 'compression-1e5.json', root / 'inputs/bank-metadata.json', bank]
    paths += [source / name for name in ('pycbc/waveform/bank.py', 'pycbc/waveform/compress.py',
              'pycbc/waveform/decompress_cpu.py', 'pycbc/waveform/decompress_torch.py')]
    paths += [source / name for name in receipt['native_modules_sha256']]
    old_source = root / 'source-v5'
    old_module_path = old_source / 'pycbc/waveform/decompress_torch.py'
    require(source_info(old_source) == dict(commit='837f38d493420043e45fb1ad210a0ccf68bacbaa', status=''),
            'Wrong frozen CUDA reference source')
    paths += [old_module_path]
    inputs = {str(path): digest(path) for path in paths}
    record = dict(state='running', passed=False, pid=os.getpid(), host=os.uname().nodename,
                  cwd=str(root), command=[sys.executable, *sys.argv], source_info=before,
                  input_sha256=inputs, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  gate='Bitwise Torch CPU/normal CPU and Torch CUDA/v5 CUDA complex64 parity at 256/512/1024 seconds',
                  expected_cases=576, completed_cases=0, native_calls=0, cuda_comparisons=0, cases=[])

    def save():
        temporary = output.with_suffix('.tmp')
        with temporary.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(output)

    save()
    started = time.monotonic()
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        import numpy as np
        import torch
        import pycbc
        from pycbc.scheme import CPUScheme, TorchScheme
        from pycbc.types.backend import backend_array
        from pycbc.waveform.bank import FilterBank
        from pycbc import waveform
        from pycbc.waveform import decompress_torch
        require(Path(pycbc.__file__).resolve().parent.parent == source, 'Wrong imported source')
        require(torch.cuda.is_available(), 'CUDA is required for this qualification')
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location('pycbc.waveform._reference_decompress_v5', old_module_path)
        old_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_module)
        original = decompress_torch._cpu_linear_interp
        original_fallback = decompress_torch._inline_interp

        def counted(*args, **kwargs):
            used = original(*args, **kwargs)
            record['native_calls'] += int(used)
            return used

        def forbid_generation(*args, **kwargs):
            raise RuntimeError('Compressed bank fell back to waveform generation')

        def compare_cuda(*args):
            out = backend_array(args[3], 'torch')
            require(out.device.type == 'cuda', 'Unexpected CPU native fallback')
            reference_output = out.clone()
            reference_args = list(args)
            reference_args[3] = reference_output
            old_module._inline_interp(*reference_args)
            result = original_fallback(*args)
            require(torch.equal(out.contiguous().view(torch.int32), reference_output.contiguous().view(torch.int32)),
                    'CUDA output differs bitwise from frozen v5 CUDA')
            record['cuda_comparisons'] += 1
            return result

        decompress_torch._cpu_linear_interp = counted
        decompress_torch._inline_interp = compare_cuda
        waveform.get_waveform_filter = forbid_generation
        metadata = json.loads((root / 'inputs/bank-metadata.json').read_text())
        expected_hashes = [str(row['template_hash']) for row in metadata['templates']]
        require(len(expected_hashes) == 96 and len(set(expected_hashes)) == 96, 'Wrong bank population')
        for length in (256, 512, 1024):
            kwargs = dict(filename=str(bank), filter_length=length * 4096 // 2 + 1,
                          delta_f=1.0 / length, dtype=np.complex64, phase_order=-1,
                          taper=None, approximant=['IMRPhenomD'], low_frequency_cutoff=30.0,
                          waveform_decompression_method='inline_linear', enable_compressed_waveforms=True)
            with CPUScheme(1):
                cpu_bank = FilterBank(**kwargs)
            with TorchScheme('cpu', num_threads=1):
                torch_bank = FilterBank(**kwargs)
            with TorchScheme('cuda:0'):
                cuda_bank = FilterBank(**kwargs)
            require(len(cpu_bank) == len(torch_bank) == len(cuda_bank) == 96, 'Bank selection changed')
            for index in range(96):
                template_hash = str(int(cpu_bank.table.template_hash[index]))
                require(template_hash == expected_hashes[index] == str(int(torch_bank.table.template_hash[index]))
                        == str(int(cuda_bank.table.template_hash[index])),
                        'Bank row identity mismatch')
                with CPUScheme(1):
                    reference = cpu_bank[index]
                    expected = reference.numpy().copy()
                    description = [len(reference), reference.delta_f, float(reference.epoch),
                                   float(reference.f_lower), float(reference.end_frequency)]
                calls_before = record['native_calls']
                with TorchScheme('cpu', num_threads=1):
                    actual = torch_bank[index]
                    values = actual.numpy()
                    observed = [len(actual), actual.delta_f, float(actual.epoch),
                                float(actual.f_lower), float(actual.end_frequency)]
                    require(description == observed, f'Metadata mismatch: {length}/{index}')
                    require(values.dtype == expected.dtype == np.dtype('complex64'), 'Wrong output precision')
                    require(np.isfinite(values).all(), 'Nonfinite compressed waveform')
                    require(np.array_equal(values.view(np.uint64), expected.view(np.uint64)),
                            f'Waveform differs bitwise: {length}/{index}')
                require(record['native_calls'] == calls_before + 1, 'Expected one native interpolation per row')
                record['cases'].append(dict(device='cpu', reference='normal CPU', length_seconds=length,
                                            index=index, template_hash=template_hash,
                                            metadata=description, output_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                                            bitwise_equal=True, native_used=True))
                record['completed_cases'] += 1
                cuda_before = record['cuda_comparisons']
                with TorchScheme('cuda:0'):
                    cuda_output = cuda_bank[index]
                    cuda_description = [len(cuda_output), cuda_output.delta_f, float(cuda_output.epoch),
                                        float(cuda_output.f_lower), float(cuda_output.end_frequency)]
                    require(cuda_description == description, 'CUDA metadata differs from normal CPU')
                    cuda_values = cuda_output.numpy()
                    require(np.isfinite(cuda_values).all(), 'Nonfinite CUDA waveform')
                    require(record['cuda_comparisons'] == cuda_before + 1, 'Missing CUDA comparison')
                record['cases'].append(dict(device='cuda', reference='frozen v5 Torch CUDA',
                                            length_seconds=length, index=index, template_hash=template_hash,
                                            metadata=cuda_description, bitwise_equal=True,
                                            output_sha256=hashlib.sha256(cuda_values.tobytes()).hexdigest()))
                record['completed_cases'] += 1
                if index % 16 == 15:
                    save()
                    print(json.dumps(dict(length=length, completed=record['completed_cases'])), flush=True)
            for opened_bank in (cpu_bank, torch_bank, cuda_bank):
                opened_bank.file.close()
            del cpu_bank, torch_bank, cuda_bank
        require(record['completed_cases'] == 576 and record['native_calls'] == record['cuda_comparisons'] == 288,
                'Incomplete qualification')
        record.update(state='complete', passed=True, returncode=0)
    except Exception as error:
        import traceback
        record.update(state='failed', passed=False, returncode=1, error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
    finally:
        record.update(source_after=source_info(source), input_sha256_after={name: digest(name) for name in inputs},
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      wall_seconds=time.monotonic() - started)
        if record['source_after'] != before or record['input_sha256_after'] != inputs:
            record.update(state='invalid-input-mutation', passed=False, returncode=1)
        save()
    print(json.dumps({key: value for key, value in record.items() if key != 'cases'}))
    return record['returncode']


if __name__ == '__main__':
    sys.exit(main())
