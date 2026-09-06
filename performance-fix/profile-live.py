"""Diagnostic-only warm cProfile/Torch profiles of unchanged public workers."""
import argparse
import cProfile
import contextlib
import io
import json
from pathlib import Path
import pstats
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--route', required=True)
p.add_argument('--batch', type=int, required=True)
p.add_argument('--threads', type=int, default=1)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(a.root))
import torch
from pycbc.filter.matchedfilter import LiveBatchMatchedFilter
from tools import bench_production_live_batch as bench

original = LiveBatchMatchedFilter.process_data
calls = 0


def profiled(self, *args, **kwargs):
    global calls
    calls += 1
    if calls == 7:
        prof = cProfile.Profile()
        with prof:
            result = original(self, *args, **kwargs)
            if 'cuda' in a.route:
                torch.cuda.synchronize()
        prof.dump_stats(str(a.out / 'python.pstats'))
        output = io.StringIO()
        pstats.Stats(prof, stream=output).strip_dirs().sort_stats('cumulative').print_stats(45)
        (a.out / 'python.txt').write_text(output.getvalue())
        return result
    if calls == 8:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if 'cuda' in a.route:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            result = original(self, *args, **kwargs)
            if 'cuda' in a.route:
                torch.cuda.synchronize()
        (a.out / 'operators.txt').write_text(prof.key_averages().table(
            sort_by='self_cpu_time_total', row_limit=40))
        prof.export_chrome_trace(str(a.out / 'trace.json'))
        return result
    return original(self, *args, **kwargs)


LiveBatchMatchedFilter.process_data = profiled
args = argparse.Namespace(route=a.route, source_root=str(a.root), batch=a.batch,
                          threads=a.threads, size=131072, num_blocks=3,
                          samples=3, warmups=1, snr_threshold=5.5,
                          cuda_device=0, seed=7101, call_surface='public')
with (a.out / 'worker.log').open('w') as log, contextlib.redirect_stdout(log):
    bench._child(args)
(a.out / 'status.json').write_text(json.dumps(dict(completed=True, calls=calls,
    scope='diagnostic attribution; instrumented worker is not a throughput measurement')))
print(a.out, flush=True)
