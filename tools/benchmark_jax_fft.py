#!/usr/bin/env python3
# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure cold and warmed PyCBC FFT round trips with explicit completion."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import time

import numpy as np

from pycbc.fft import fft, ifft, backend_cpu
from pycbc.scheme import CPUScheme, JAXScheme
from pycbc.types import Array, zeros
from pycbc.types.backend import backend_array


def benchmark(processing_scheme='jax:cpu', dtype='complex64', length=65536,
              repeats=20, fft_backend='numpy', reference_operations=(), profile=None):
    if length < 1 or repeats < 1:
        raise ValueError('Length and repeats must be positive')
    is_jax = processing_scheme.startswith('jax:')
    if not is_jax and (processing_scheme != 'cpu' or reference_operations or profile):
        raise ValueError('Native CPU mode does not accept JAX controls or profiling')
    context = (JAXScheme(processing_scheme[4:], reference_operations=reference_operations)
               if is_jax else CPUScheme())
    rng = np.random.default_rng(1784)
    values = (rng.normal(size=length) + 1j * rng.normal(size=length)).astype(dtype)
    previous = backend_cpu.cpu_backend
    try:
        backend_cpu.set_backend([fft_backend])
        if backend_cpu.cpu_backend != fft_backend:
            raise ValueError('Requested CPU FFT backend is unavailable')
        with context:
            source = Array(values)
            transformed, result = zeros(length, dtype=dtype), zeros(length, dtype=dtype)

            def roundtrip():
                fft(source, transformed)
                ifft(transformed, result)
                if is_jax:
                    backend_array(result, 'jax').block_until_ready()

            start = time.perf_counter()
            roundtrip()
            first = time.perf_counter() - start
            # Bare PyCBC arrays use an unnormalized inverse FFT.
            restored = result.numpy() / length
            tolerance = 5e-5 if dtype == 'complex64' else 1e-12
            np.testing.assert_allclose(restored, values, rtol=tolerance, atol=tolerance)
            tracer = nullcontext()
            if profile:
                import jax
                tracer = jax.profiler.trace(str(profile))
            samples = []
            with tracer:
                for _ in range(repeats):
                    start = time.perf_counter()
                    roundtrip()
                    samples.append(time.perf_counter() - start)
            record = dict(processing_scheme=processing_scheme, dtype=dtype, length=length,
                          repeats=repeats, cpu_fft_backend=fft_backend,
                          reference_operations=list(reference_operations), first_seconds=first,
                          median_seconds=float(np.median(samples)), samples_seconds=samples,
                          max_roundtrip_error=float(np.max(np.abs(restored - values))),
                          numpy_version=np.__version__, architecture=platform.machine())
            if is_jax:
                import jax
                record.update(jax_version=jax.__version__, device_kind=context.jax_device.device_kind)
            return record
    finally:
        backend_cpu.cpu_backend = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processing-scheme', default='jax:cpu')
    parser.add_argument('--dtype', choices=['complex64', 'complex128'], default='complex64')
    parser.add_argument('--length', type=int, default=65536)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--fft-backend', choices=['numpy', 'fftw', 'mkl'], default='numpy')
    parser.add_argument('--reference-operation', choices=['fft', 'ifft'], action='append', default=[])
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    record = benchmark(args.processing_scheme, args.dtype, args.length, args.repeats,
                       args.fft_backend, tuple(args.reference_operation), args.profile)
    text = json.dumps(record, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
