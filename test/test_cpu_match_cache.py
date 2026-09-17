"""CPU scratch reuse must survive changes in CPU scheme/thread count."""
import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter import matchedfilter
from pycbc.types import FrequencySeries


def test_match_reuses_cpu_scratch_across_contexts(monkeypatch):
    monkeypatch.setattr(matchedfilter, "_snr", None)
    monkeypatch.setattr(matchedfilter, "_snr_scheme_key", None)
    monkeypatch.setattr(scheme.Scheme, "_single", None)
    vector = FrequencySeries(np.ones(9, dtype=np.complex64), delta_f=.5)
    matchedfilter.match(vector, vector)
    original = matchedfilter._snr
    context = scheme.CPUScheme(1)
    for threads in (1, 2):
        context.num_threads = threads
        with context:
            value, _ = matchedfilter.match(vector, vector)
            assert value == pytest.approx(1., abs=1e-6)
            assert matchedfilter._snr is original


def test_match_keeps_torch_scratch_separate(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(matchedfilter, "_snr", None)
    monkeypatch.setattr(matchedfilter, "_snr_scheme_key", None)
    cpu_vector = FrequencySeries(np.ones(9, dtype=np.complex64), delta_f=.5)
    matchedfilter.match(cpu_vector, cpu_vector)
    original = matchedfilter._snr
    with scheme.TorchScheme("cpu"):
        vector = FrequencySeries(np.ones(9, dtype=np.complex64), delta_f=.5)
        value, _ = matchedfilter.match(vector, vector)
        assert value == pytest.approx(1., abs=1e-6)
        assert matchedfilter._snr is not original
        assert matchedfilter._snr.backend == "torch"
    matchedfilter.match(cpu_vector, cpu_vector)
    assert matchedfilter._snr.backend == "numpy"
