import types
import numpy as np

from pycbc import scheme
from pycbc.filter import matchedfilter
from pycbc.types import Array


def _build_cpu_batch(rows, threshold=2.5, abort_threshold=None):
    rows = np.asarray(rows, dtype=np.complex64)
    template_count, template_size = rows.shape
    output = Array(rows.reshape(-1))
    templates = []
    for index in range(template_count):
        params = np.array(
            [(20.0 + index,)], dtype=[("mass1", np.float32)]
        )[0]
        template = types.SimpleNamespace(
            delta_f=0.25,
            id=10 + index,
            params=params,
            out=output[index * template_size:(index + 1) * template_size],
            sigmasq=lambda _psd: 1.0,
        )
        templates.append(template)

    mid = ("test", template_count)
    calls = {"correlate": 0, "ifft": 0}
    batch = object.__new__(matchedfilter.LiveBatchMatchedFilter)
    batch.block_id = 0
    batch.tgroups = [templates]
    batch.chunk_tsamples = [template_size]
    batch.mids = [mid]
    batch.out_mem = {mid: output}
    batch.corr = [types.SimpleNamespace(
        execute=lambda _data: calls.__setitem__(
            "correlate", calls["correlate"] + 1
        )
    )]
    batch.ifts = {mid: types.SimpleNamespace(
        execute=lambda: calls.__setitem__("ifft", calls["ifft"] + 1)
    )}
    batch.data = types.SimpleNamespace(
        overwhitened_data=lambda _delta_f: types.SimpleNamespace(psd=object()),
        trim_padding=1,
        blocksize=4,
        sample_rate=1,
        start_time=100.0,
    )
    batch.snr_threshold = threshold
    batch.snr_abort_threshold = abort_threshold
    return batch, templates, calls


def _legacy_cpu_peaks(templates, seg=slice(3, 7)):
    peaks = []
    for template in templates:
        relative_index = template.out[seg].abs_arg_max()
        peaks.append((relative_index,
                      template.out[relative_index + seg.start]))
    return peaks


def test_cpu_batch_peak_values_direct(monkeypatch):
    monkeypatch.setenv("PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK", "1")
    rng = np.random.default_rng(42)
    template_count = 16
    template_size = 128
    seg = slice(20, 100)

    raw = (
        rng.standard_normal((template_count, template_size))
        + 1j * rng.standard_normal((template_count, template_size))
    ).astype(np.complex64)
    arr = Array(raw.reshape(-1))

    indices, peaks = matchedfilter._cpu_batch_peak_values(
        arr, template_count, template_size, seg
    )

    assert indices is not None
    assert peaks is not None
    assert len(indices) == template_count
    assert len(peaks) == template_count

    for i in range(template_count):
        row_slice = raw[i, 20:100]
        mags = row_slice.real**2 + row_slice.imag**2
        expected_idx = np.argmax(mags)
        assert indices[i] == expected_idx
        assert peaks[i] == row_slice[expected_idx]


def test_cpu_batch_peak_values_fallbacks(monkeypatch):
    monkeypatch.setenv("PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK", "1")
    arr = np.ones((4, 32), dtype=np.complex64)
    # Single template fallback
    assert matchedfilter._cpu_batch_peak_values(
        arr, 1, 32, slice(0, 10)
    ) is None
    # Step != 1 fallback
    assert matchedfilter._cpu_batch_peak_values(
        arr, 4, 32, slice(0, 10, 2)
    ) is None
    # Wrong size fallback
    assert matchedfilter._cpu_batch_peak_values(
        arr, 4, 16, slice(0, 10)
    ) is None
    # Wrong dtype fallback
    arr_f32 = np.ones((4, 32), dtype=np.float32)
    assert matchedfilter._cpu_batch_peak_values(
        arr_f32, 4, 32, slice(0, 10)
    ) is None


def test_cpu_live_batch_peak_reduction_parity(monkeypatch):
    monkeypatch.setenv("PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK", "1")
    rows = np.array([
        [99, 0, 0, 0, 0, 0, 0, 99],
        [99, 0, 0, 1, 3, -3, 2, 99],
        [99, 0, 0, 1, 2, np.nan, 3, 99],
        [99, 0, 0, 1, 2, 3, -4j, 99],
    ], dtype=np.complex64)

    with scheme.CPUScheme():
        batch, templates, calls = _build_cpu_batch(rows, threshold=2.5)
        legacy_peaks = _legacy_cpu_peaks(templates)

        # Run with vectorized CPU peak finding
        result, veto_info = batch._process_batch()

    assert legacy_peaks[0][0] == 0
    assert legacy_peaks[1][0] == 1
    assert legacy_peaks[3][0] == 3
    assert legacy_peaks[0][1] == 0
    assert legacy_peaks[1][1] == 3
    assert legacy_peaks[3][1] == -4j

    assert calls == {"correlate": 1, "ifft": 1}
    assert batch.block_id == 1

    np.testing.assert_array_equal(result["template_id"], [11, 12, 13])
    np.testing.assert_array_equal(result["mass1"], [21, 22, 23])

    expected_peaks = np.asarray(
        [peak for _index, peak in legacy_peaks[1:]], dtype=np.complex64
    )
    np.testing.assert_allclose(
        result["snr"], np.abs(expected_peaks), equal_nan=True
    )
    np.testing.assert_allclose(
        result["coa_phase"], np.angle(expected_peaks), equal_nan=True
    )
    expected_indices = [index for index, _peak in legacy_peaks[1:]]
    np.testing.assert_array_equal(
        result["end_time"], 100 + np.asarray(expected_indices)
    )
    np.testing.assert_array_equal(result["sigmasq"], [1, 1, 1])
    assert [info[2] for info in veto_info] == [
        3 + index for index in expected_indices
    ]
    assert all(info[0].dtype == np.dtype(np.complex128) for info in veto_info)
    assert veto_info[0][0][0] == 3
    if np.isnan(legacy_peaks[2][1]):
        assert np.isnan(veto_info[1][0][0])
    else:
        assert veto_info[1][0][0] == legacy_peaks[2][1]
    assert veto_info[2][0][0] == -4j
