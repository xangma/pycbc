"""Original numerical routes remain reachable through batched searches."""

import math
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc import scheme
from pycbc.filter.matchedfilter import MatchedFilterControl
from pycbc.filter.matchedfilter_jax import (
    JAXMatchedFilterControl, process_batch_inspiral_jax,
)
from pycbc.types import Array, FrequencySeries, zeros


from pycbc.waveform.bank_jax import TemplateBatchList
from pycbc.waveform.bank import sigma_cached


def _control(control_type, strain, n_time=128):
    segment = FrequencySeries(strain, delta_f=1.0)
    segment.analyze = slice(9, n_time - 11)
    return control_type(3.0, 60.0, 0.1, n_time, 1.0, np.complex64,
                        [segment], zeros(n_time, dtype=np.complex64),
                        use_cluster=True, cluster_function="symmetric")


@pytest.mark.parametrize("count", [1, 3])
def test_batched_filter_composes_original_correlation_ifft_and_cluster(count):
    rng = np.random.default_rng(777)
    templates = (rng.normal(size=(count, 65))
                 + 1j * rng.normal(size=(count, 65))).astype(np.complex64)
    strain = (rng.normal(size=65) + 1j * rng.normal(size=65)).astype(np.complex64)
    sigmasqs = [1.234567890123 + row for row in range(count)]
    with scheme.CPUScheme():
        control = _control(MatchedFilterControl, strain)
        expected = []
        for values, sigmasq in zip(templates, sigmasqs):
            control.htilde[:len(values)] = Array(values)
            result = control.matched_filter_and_cluster(0, sigmasq, 8)
            expected.append(tuple(value.copy() if hasattr(value, "copy")
                                  else value for value in result))
    with scheme.JAXScheme("cpu", reference_operations=(
            "correlate", "ifft", "threshold_cluster")):
        control = _control(JAXMatchedFilterControl, strain)
        actual = control.batched_matched_filter_and_cluster(
            0, templates, sigmasqs, 8)
        for result, reference in zip(actual, expected):
            assert len(result[3]) > 0
            assert result[1] == reference[1]
            for index in (0, 2, 3, 4):
                got, wanted = np.asarray(result[index]), np.asarray(reference[index])
                assert got.dtype == wanted.dtype
                assert got.tobytes() == wanted.tobytes()






def test_inspiral_batch_preserves_native_normalization_precision():
    rng = np.random.default_rng(173)
    values = (rng.normal(size=257)
              + 1j * rng.normal(size=257)).astype(np.complex64)
    spectrum = rng.uniform(.5, 2, 257).astype(np.float32)
    with scheme.CPUScheme():
        template = FrequencySeries(values, delta_f=.5)
        template.approximant = "IMRPhenomD"
        template.f_lower = 2
        template.min_f_lower = 1.5
        template.end_frequency = 90
        template.params = SimpleNamespace()
        template.sigmasq = MethodType(sigma_cached, template)
        expected = template.sigmasq(FrequencySeries(spectrum, delta_f=.5))
    assert float(np.float32(expected)) != expected

    class Bank:
        def __len__(self):
            return 1

        def get_batch(self, indices):
            assert indices == [0]
            return TemplateBatchList([template])

    captured = []
    events = []
    with scheme.JAXScheme("cpu", reference_operations=(
            "squared_norm", "divide", "inner", "correlate", "ifft",
            "threshold_cluster")):
        segment = FrequencySeries(values, delta_f=.5)
        segment.analyze = slice(9, 501)
        segment.cumulative_index = 0
        segment.psd = FrequencySeries(spectrum, delta_f=.5)
        control = JAXMatchedFilterControl(
            1.5, 90, .1, 512, .5, np.complex64, [segment],
            zeros(512, dtype=np.complex64), use_cluster=True,
            cluster_function="symmetric")

        def filter_batch(segment_number, templates, sigmasqs, window, **kwargs):
            result = control.batched_matched_filter_and_cluster(
                segment_number, templates, sigmasqs, window, **kwargs)
            captured.append((np.asarray(sigmasqs).copy(), result[0][1]))
            return result

        process_batch_inspiral_jax(
            SimpleNamespace(add_template_events_direct=lambda *args, **kwargs:
                            events.append((args, kwargs))),
            Bank(), [0], [segment],
            SimpleNamespace(batched_matched_filter_and_cluster=filter_batch),
            None, 8)
    assert len(events) == 1
    assert len(events[0][0][1]) > 0
    assert events[0][1]["sigmasq"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(events[0][1]["sigmasq"], np.float32(expected))
    assert len(captured) == 1
    sigmasqs, norm = captured[0]
    assert sigmasqs.dtype == np.dtype(np.float64)
    assert sigmasqs[0] == expected
    assert norm == 2.0 / math.sqrt(expected)
