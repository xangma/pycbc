"""NumPy point chi-square used by the JAX backend on CPU devices.

Native CPUScheme dispatch retains its existing compiled implementation.
"""

import numpy as np


def shift_sum(corr, points, bins, n_time=None, base_k=0):
    """Sum squared bin amplitudes using double-precision Fourier phases.

    ``bins`` contains absolute frequency indices. ``base_k`` identifies the
    first frequency of a cropped correlation, whose full FFT length must be
    supplied as ``n_time``. Return the correlation's real precision, but form
    phases, complex sums, and accumulated power in double precision.

    Frequencies are processed in bounded chunks for each selected sample;
    storage does not grow with the number of samples or the FFT length.
    """
    corr = np.asarray(corr)
    points = np.asarray(points, dtype=np.int64)
    bins = np.asarray(bins, dtype=np.int64)
    n_time = len(corr) if n_time is None else int(n_time)
    output = np.empty(len(points), dtype=corr.real.dtype)
    chunk_size = 65536
    for index, point in enumerate(points):
        total_power = 0.0
        point = point % n_time
        for start, end in zip(bins[:-1], bins[1:]):
            amplitude = 0j
            # Cropped correlations represent zero power outside their stored
            # frequency interval; retain the original bins, including empties.
            start = max(start, base_k)
            end = min(end, base_k + len(corr))
            for offset in range(start, end, chunk_size):
                stop = min(offset + chunk_size, end)
                frequencies = np.arange(offset, stop, dtype=np.int64)
                cycles = (frequencies * point) % n_time
                phase = np.exp((2j * np.pi / n_time) * cycles)
                amplitude += np.sum(
                    corr[offset-base_k:stop-base_k] * phase,
                    dtype=np.complex128,
                )
            total_power += amplitude.real ** 2 + amplitude.imag ** 2
        output[index] = total_power
    return output
