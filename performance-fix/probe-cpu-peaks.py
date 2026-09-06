"""Compare exact squared-magnitude expressions outside the live pipeline."""
import json
import statistics
import time
import torch

torch.manual_seed(7101)
for threads in [1, 4]:
    torch.set_num_threads(threads)
    for batch in [32, 1024]:
        values = torch.randn(batch, 131072, dtype=torch.complex64)[:, 12288:126976]
        functions = {
            'original': lambda: torch.view_as_real(values).square().sum(dim=-1),
            'components': lambda: values.real.square() + values.imag.square(),
        }
        expected = functions['original']()
        actual = functions['components']()
        assert torch.equal(expected, actual)
        del expected, actual
        for name, function in functions.items():
            function()
            samples=[]
            for _ in range(5):
                start=time.perf_counter()
                result=function()
                indices=torch.argmax(result,dim=-1)
                samples.append(time.perf_counter()-start)
                del result, indices
            print(json.dumps(dict(batch=batch,threads=threads,method=name,
                median_seconds=statistics.median(samples),samples=samples,
                exact_equal=True)),flush=True)
        del values
