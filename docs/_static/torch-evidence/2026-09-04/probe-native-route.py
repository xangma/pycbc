"""Untimed correlation confirmation; exclude its output from timing artifacts."""
import argparse
import json
import sys
from tools import bench_production_live_batch as bench
from pycbc.filter import matchedfilter_torch

source, route, threads, batch = sys.argv[1:]
name = ("_try_cuda_native_batch_correlate" if "cuda" in route
        else "_try_cpu_native_batch_correlate")
original = getattr(matchedfilter_torch, name)
counts = {"attempts": 0, "native_successes": 0, "fallbacks": 0}


def observe(batch, data):
    result = original(batch, data)
    counts["attempts"] += 1
    counts["native_successes"] += int(result)
    counts["fallbacks"] += int(not result)
    return result


setattr(matchedfilter_torch, name, observe)
bench._child(argparse.Namespace(
    source_root=source, route=route, threads=int(threads), cuda_device=0,
    batch=int(batch), size=131072, num_blocks=3, samples=3, warmups=0,
    snr_threshold=5.5, seed=7101, call_surface="public",
))
print("ROUTE_PROBE=" + json.dumps({
    "component": "batch_correlation", "route": route,
    "threads": int(threads), "batch": int(batch), **counts,
}, sort_keys=True))
assert counts["native_successes"] > 0, counts
assert counts["fallbacks"] == 0, counts
