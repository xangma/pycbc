"""Destructive negative control for the private graph allocation owner only."""
import fcntl
import json
import os
from pathlib import Path

import torch
from pycbc.filter._torch_cuda_graph import _GraphEntry


def exercise(protected):
    torch.cuda.synchronize()
    allocation = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    count = 1048576
    with torch.cuda.stream(allocation):
        scratch = torch.ones(count, dtype=torch.bool, device='cuda')
        raw = torch.zeros((), dtype=torch.float32, device='cuda')
        squared = torch.empty_like(raw)
    before = scratch.data_ptr()
    entry = _GraphEntry((('synthetic',), (scratch,), ()), ((), ()), raw, squared, consumer)
    observed = torch.empty_like(scratch)
    consumer.wait_stream(allocation)
    if protected:
        entry.record_stream(consumer)
    event = torch.cuda.Event()
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(500000000)
        observed.copy_(scratch)
        event.record()
    assert not event.query(), 'Probe did not reach pending work'
    with torch.cuda.stream(allocation):
        scratch.resize_(count * 2)
        pending_at_mutation = not event.query()
        competing = [torch.zeros(count, dtype=torch.bool, device='cuda') for _ in range(8)]
    assert pending_at_mutation, 'GPU work finished before mutation'
    event.synchronize()
    torch.cuda.synchronize()
    correct = bool(observed.all().item())
    reused = before in [tensor.data_ptr() for tensor in competing]
    entry.close()
    return dict(protected=protected, pending_at_mutation=pending_at_mutation,
                original_pointer_reused=reused, sentinel_preserved=correct)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    lock = open('/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = [exercise(True), exercise(False)]
    result = dict(pid=os.getpid(), rows=rows)
    (root / 'pending-allocation-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    assert rows[0]['sentinel_preserved'] and not rows[0]['original_pointer_reused']
    assert not rows[1]['sentinel_preserved'] and rows[1]['original_pointer_reused']
