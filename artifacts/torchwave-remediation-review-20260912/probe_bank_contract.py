"""Read-only source review reproducer; creates and removes a temporary bank."""
import json
import os
import tempfile

import h5py
import numpy as np
import torch

from pycbc.waveform.bank import FilterBank


fd, path = tempfile.mkstemp(suffix=".hdf")
os.close(fd)
try:
    with h5py.File(path, "w") as bank_file:
        data = {
            "mass1": [1.4, 1.8], "mass2": [1.2, 1.4],
            "spin1z": [0.0, 0.0], "spin2z": [0.0, 0.0],
            "f_lower": [30.0, 30.0],
        }
        for key, values in data.items():
            bank_file[key] = np.asarray(values, dtype=np.float64)
        bank_file.attrs["parameters"] = list(data)

    common = dict(filter_length=2049, delta_f=0.25, dtype=np.complex64,
                  approximant="TaylorF2", enable_torchwave=True)
    bank0 = FilterBank(path, phase_order=0, **common)
    bank7 = FilterBank(path, phase_order=7, **common)
    batch0, _ = bank0.get_batch_tensor([0])
    batch7, _ = bank7.get_batch_tensor([0])
    reference0 = bank0[0].numpy().copy()
    reference7 = bank7[0].numpy().copy()
    requested128, _ = bank7.get_batch_tensor([0], dtype=torch.complex128)
    automatic = FilterBank(path, **{
        key: value for key, value in common.items() if key != "enable_torchwave"
    })
    print(json.dumps({
        "probe": "phase_order_and_dtype",
        "torchwave_batch_equal_for_phase_order_0_vs_7": bool(
            torch.equal(batch0, batch7)),
        "reference_relative_l2_change": float(
            np.linalg.norm(reference0 - reference7) / np.linalg.norm(reference7)),
        "requested_dtype": "torch.complex128",
        "returned_dtype": str(requested128.dtype),
        "default_can_use_torchwave": automatic.can_use_torchwave(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }, indent=2))
finally:
    os.remove(path)
