"""Execute the two standalone examples against the matching built checkout."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pycbc
import torch
from pycbc.scheme import TorchScheme
from pycbc.types import TimeSeries
from pycbc.waveform import get_fd_waveform_batch

AUDIT = Path('/private/tmp/pycbc-torch-doc-audit-20260908')
BUILT = Path('/private/tmp/pycbc-torch-publication-20260908')
OUTPUT = Path(__file__).resolve().parent
files = subprocess.check_output(
    ['git', 'ls-files', 'pycbc'], cwd=AUDIT, text=True
).splitlines()
assert all((AUDIT / f).read_bytes() == (BUILT / f).read_bytes() for f in files)
before_threads = torch.get_num_threads()
with TorchScheme('cpu', num_threads=4):
    series = TimeSeries([1.0, 2.0, 3.0], delta_t=0.25)
    squared = series * series
    np.testing.assert_array_equal(squared.numpy(), [1.0, 4.0, 9.0])
    assert torch.get_num_threads() == 4
assert torch.get_num_threads() == before_threads

with TorchScheme('cpu'):
    batch = get_fd_waveform_batch(
        'TaylorF2',
        mass1=[1.4, 1.5], mass2=1.3,
        f_lower=[20.0, 24.0], f_final=128.0, delta_f=1.0,
    )
    assert batch.hplus.shape == batch.hcross.shape
    assert batch.hplus.shape[0] == 2
    assert torch.isfinite(batch.hplus).all() and torch.isfinite(batch.hcross).all()
    assert batch.first_bins.tolist() == [20, 24]
    assert batch.end_bins.tolist() == [129, 129]
    for i, (start, end) in enumerate(zip(batch.first_bins.tolist(), batch.end_bins.tolist())):
        assert torch.count_nonzero(batch.hplus[i, :start]) == 0
        assert torch.count_nonzero(batch.hplus[i, end:]) == 0

result = {
    'quickstart': 'passed',
    'thread_restoration': 'passed',
    'waveform_batch': 'passed',
    'shape': list(batch.hplus.shape),
    'first_bins': batch.first_bins.tolist(),
    'end_bins': batch.end_bins.tolist(),
    'torch': torch.__version__,
    'device': str(batch.hplus.device),
    'pycbc_source': str(Path(pycbc.__file__).resolve()),
    'tracked_runtime_files_byte_identical': len(files),
    'initial_attempts': [
        'Unbuilt audit checkout lacked native extensions; used matching built editable checkout.',
        'An assertion treated the TorchArrayData wrapper dtype as torch.dtype; corrected assertion to use the public numpy() conversion.',
    ],
}
(OUTPUT / 'quickstart-validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
