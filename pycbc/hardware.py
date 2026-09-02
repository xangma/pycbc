"""Hardware inspection and cache hierarchy detection utilities for PyCBC."""

import ctypes
import glob
import math
import os
import re
import sys


def _parse_cache_size_string(size_str):
    """Parse a size string (e.g., '32M', '16384K', '1G') into integer bytes."""
    clean = str(size_str).strip().upper()
    if not clean:
        return 0
    if clean.endswith("K") or clean.endswith("KB"):
        digits = re.sub(r"[^0-9]", "", clean)
        return int(digits) * 1024 if digits else 0
    if clean.endswith("M") or clean.endswith("MB"):
        digits = re.sub(r"[^0-9]", "", clean)
        return int(digits) * 1024 * 1024 if digits else 0
    if clean.endswith("G") or clean.endswith("GB"):
        digits = re.sub(r"[^0-9]", "", clean)
        return int(digits) * 1024 * 1024 * 1024 if digits else 0
    digits = re.sub(r"[^0-9]", "", clean)
    return int(digits) if digits else 0


def get_gpu_l2_cache_size(device_id=0):
    """Return the L2 cache size in bytes for the specified CUDA GPU device.

    Queries through PyTorch device properties, the CUDA Driver API via ctypes,
    and microarchitecture lookup tables before falling back to a safe default.
    """
    # 1. Direct PyTorch property lookup if available in current runtime
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_id)
            if hasattr(props, "l2_cache_size") and props.l2_cache_size:
                return int(props.l2_cache_size)
    except Exception:
        pass

    # 2. Query CUDA Driver API (CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38)
    for libname in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            nvcuda = ctypes.CDLL(libname)
            if nvcuda.cuInit(0) == 0:
                dev = ctypes.c_int()
                if nvcuda.cuDeviceGet(ctypes.byref(dev), int(device_id)) == 0:
                    val = ctypes.c_int()
                    if nvcuda.cuDeviceGetAttribute(
                        ctypes.byref(val), 38, dev
                    ) == 0:
                        if val.value > 0:
                            return int(val.value)
        except Exception:
            pass

    # 3. Microarchitecture fallback based on compute capability
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_id)
            arch = (props.major, props.minor)
            total_mem = getattr(props, "total_memory", 0)
            if arch >= (10, 0):  # Blackwell
                return 128 * 1024 * 1024
            elif arch >= (9, 0):  # Hopper
                return 50 * 1024 * 1024
            elif arch >= (8, 9):  # Ada Lovelace (RTX 4090 / 4080)
                return 72 * 1024 * 1024
            elif arch >= (8, 0):  # Ampere (A100 vs GA102 / RTX 3090)
                return 40 * 1024 * 1024 if total_mem > 30 * 1024**3 else 6 * 1024 * 1024
            elif arch >= (7, 0):  # Volta / Turing
                return 6 * 1024 * 1024
    except Exception:
        pass

    # Safe fallback (6 MB)
    return 6 * 1024 * 1024


def get_cpu_l3_cache_size(per_ccx=True):
    """Return the CPU L3/L2 cache size in bytes.

    Parameters
    ----------
    per_ccx : bool, default True
        If True, returns the L3 cache size per physical Core Complex / CCD
        (e.g., 16 MB or 32 MB) to prevent cross-CCX cache thrashing in
        multi-core architectures like AMD Zen / Threadripper. If False,
        returns the total system L3 cache size across all sockets.
    """
    # 1. Linux sysfs inspection (/sys/devices/system/cpu/cpu*/cache/index3/)
    try:
        paths = sorted(glob.glob("/sys/devices/system/cpu/cpu*/cache/index3/size"))
        if not paths:
            paths = sorted(glob.glob("/sys/devices/system/cpu/cpu*/cache/index2/size"))
        if paths:
            l3_caches = {}
            for p in paths:
                dirpath = os.path.dirname(p)
                id_file = os.path.join(dirpath, "id")
                shared_file = os.path.join(dirpath, "shared_cpu_list")
                if os.path.exists(id_file):
                    cid = open(id_file).read().strip()
                elif os.path.exists(shared_file):
                    cid = open(shared_file).read().strip()
                else:
                    cid = p
                if cid not in l3_caches:
                    with open(p) as f:
                        l3_caches[cid] = _parse_cache_size_string(f.read())
            if l3_caches:
                if per_ccx:
                    return next(iter(l3_caches.values()))
                return sum(l3_caches.values())
    except Exception:
        pass

    # 2. macOS sysctl inspection
    if sys.platform == "darwin":
        import subprocess
        for key in ("hw.l3cachesize", "hw.perflevel0.l2cachesize", "hw.l2cachesize"):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", key],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if out and int(out) > 0:
                    return int(out)
            except Exception:
                pass

    # Safe default: 32 MB per CCX, 128 MB system total
    return 32 * 1024 * 1024 if per_ccx else 128 * 1024 * 1024


def get_optimal_batch_maxelements(is_cuda=False, device_id=0, safety_factor=0.85):
    """Compute the optimal maxelements chunk size for cache-resident batched filtering.

    Parameters
    ----------
    is_cuda : bool
        Whether the execution scheme is running on a CUDA GPU.
    device_id : int, default 0
        CUDA device ordinal.
    safety_factor : float, default 0.85
        Fraction of cache capacity allowed for the correlation and transform buffers.

    Returns
    -------
    int
        Optimal maximum number of complex64 elements per batch chunk.
    """
    if "PYCBC_BATCH_MAXELEMENTS" in os.environ and os.environ["PYCBC_BATCH_MAXELEMENTS"].strip():
        try:
            return int(os.environ["PYCBC_BATCH_MAXELEMENTS"].strip())
        except ValueError:
            pass

    if is_cuda:
        cache_bytes = get_gpu_l2_cache_size(device_id)
        # For complex64 (8 bytes/element), working set across correlation & IFFT is ~8B/elem.
        elements = int(cache_bytes * safety_factor / 8)
        # Bound chunk to power of 2 between 2**19 (512k) and 2**24 (16M)
        power = int(math.floor(math.log2(max(1, elements))))
        return 2 ** max(19, min(24, power))
    else:
        # For CPU, scale against per-CCX L3 cache to avoid cache evictions
        cache_bytes = get_cpu_l3_cache_size(per_ccx=True)
        elements = int(cache_bytes * safety_factor / 8)
        power = int(math.floor(math.log2(max(1, elements))))
        # Return at least 2**21 elements (~2M) up to 2**27 (~134M)
        return 2 ** max(21, min(27, power))


def get_optimal_batch_tile_size(
    transform_size,
    is_cuda=False,
    device_id=0,
    safety_factor=0.85,
    min_tile=8,
    max_tile=64,
):
    """Compute optimal batch tile count to ensure active transforms fit in cache.

    Parameters
    ----------
    transform_size : int
        Number of complex frequency/time domain samples (e.g., 131,072).
    is_cuda : bool, default False
        Whether running on a CUDA GPU.
    device_id : int, default 0
        CUDA device ordinal.
    safety_factor : float, default 0.85
        Fraction of cache capacity dedicated to active working buffers.
    min_tile : int, default 8
        Minimum tile size.
    max_tile : int, default 64
        Maximum tile size.

    Returns
    -------
    int
        Optimal number of templates/transforms per execution tile.
    """
    if "PYCBC_BATCH_TILE_SIZE" in os.environ and os.environ["PYCBC_BATCH_TILE_SIZE"].strip():
        try:
            val = int(os.environ["PYCBC_BATCH_TILE_SIZE"].strip())
            if val > 0:
                return val
        except ValueError:
            pass

    if is_cuda:
        cache_bytes = get_gpu_l2_cache_size(device_id)
    else:
        cache_bytes = get_cpu_l3_cache_size(per_ccx=True)

    # In/out buffer pair per template occupies: 2 * transform_size * 8 bytes
    bytes_per_template = 2 * int(transform_size) * 8
    if bytes_per_template <= 0:
        return max_tile

    allowed_bytes = cache_bytes * safety_factor
    optimal = int(allowed_bytes / bytes_per_template)

    if optimal <= min_tile:
        return min_tile

    power = 2 ** int(round(math.log2(optimal)))
    return max(min_tile, min(max_tile, power))


def get_cpu_cores_per_numa_node():
    """Return the number of CPU cores/threads sharing a single L3 cache/NUMA node."""
    import multiprocessing

    # 1. Linux sysfs inspection
    try:
        shared_list = (
            "/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list"
        )
        if not os.path.exists(shared_list):
            shared_list = (
                "/sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_list"
            )
        if os.path.exists(shared_list):
            with open(shared_list) as f:
                content = f.read().strip()
            total = 0
            for part in content.split(","):
                if "-" in part:
                    start, end = part.split("-")
                    total += int(end) - int(start) + 1
                elif part.isdigit():
                    total += 1
            if total > 0:
                return total
    except Exception:
        pass

    # 2. Darwin sysctl inspection
    if sys.platform == "darwin":
        import subprocess

        for key in ("hw.perflevel0.logicalcpu", "hw.logicalcpu"):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", key],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if out and int(out) > 0:
                    return int(out)
            except Exception:
                pass

    return min(8, multiprocessing.cpu_count())


def get_optimal_1d_fft_threads(transform_size, requested_threads=None):
    """Compute the optimal thread count for a single 1D FFT to prevent cache thrashing.

    Parameters
    ----------
    transform_size : int
        Number of complex frequency/time domain samples (e.g., 131,072).
    requested_threads : int, optional
        User-requested or scheme-active thread count.

    Returns
    -------
    int
        Optimal thread team size for the 1D transform.
    """
    if requested_threads is None:
        import multiprocessing

        requested_threads = multiprocessing.cpu_count()
    requested_threads = int(max(1, requested_threads))

    # Working set for single complex64 in/out pair: 2 * transform_size * 8 bytes
    working_set_bytes = 2 * int(transform_size) * 8
    ccx_cache_bytes = get_cpu_l3_cache_size(per_ccx=True)

    if working_set_bytes <= ccx_cache_bytes:
        # Fits inside a single CCD/L3 cache domain: bound to cores in this domain
        cores_in_node = get_cpu_cores_per_numa_node()
        return max(1, min(requested_threads, cores_in_node))

    return requested_threads

