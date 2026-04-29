import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed


pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


def test_power_chisq_at_points_torch_matches_cpu_and_stays_real():
    rng = np.random.default_rng(1234)
    corr = (rng.normal(size=128) + 1j * rng.normal(size=128)).astype(np.complex64)
    snr = (rng.normal(size=5) + 1j * rng.normal(size=5)).astype(np.complex64)
    indices = np.array([0, 3, 7, 10, 17], dtype=np.int64)
    bins = np.array([4, 12, 24, 40, 80], dtype=np.int64)
    snr_norm = 0.37

    cpu_corr = FrequencySeries(corr.copy(), delta_f=0.1)
    cpu = power_chisq_at_points_from_precomputed(
        cpu_corr, snr.copy(), snr_norm, bins, indices
    )

    with scheme.TorchScheme("cpu"):
        torch_corr = FrequencySeries(corr.copy(), delta_f=0.1)
        chisq = power_chisq_at_points_from_precomputed(
            torch_corr, snr.copy(), snr_norm, bins, indices
        )

    assert chisq.kind == "real"
    np.testing.assert_allclose(chisq.numpy(), cpu, rtol=1e-5, atol=1e-5)
