import logging
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pycbc
from pycbc import scheme
from pycbc.filter import matchedfilter
from pycbc.types import Array, FrequencySeries
from pycbc.types.array_torch import TorchArrayData


if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without Torch support", allow_module_level=True)


def _available_torch_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(params=_available_torch_devices())
def torch_device(request):
    return request.param


def _build_batch(rows, threshold, abort_threshold=None):
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


def _legacy_torch_peaks(templates):
    peaks = []
    for template in templates:
        relative_index = template.out[3:7].abs_arg_max()
        peaks.append((relative_index, template.out[relative_index + 3]))
    return peaks


def _run_without_scalar_sync(batch, monkeypatch):
    copies = []
    original_cpu = torch.Tensor.cpu

    def count_bulk_copy(tensor, *args, **kwargs):
        copies.append((tuple(tensor.shape), tensor.dtype, tensor.device.type))
        return original_cpu(tensor, *args, **kwargs)

    def reject_scalar_sync(_tensor, *_args, **_kwargs):
        raise AssertionError("batched peak extraction synchronized a scalar")

    def reject_array_copy(_self):
        raise AssertionError("batched peak extraction copied an output array")

    def reject_legacy_peak(_self):
        raise AssertionError("batched peak extraction used scalar argmax")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", count_bulk_copy)
        patch.setattr(torch.Tensor, "item", reject_scalar_sync)
        patch.setattr(TorchArrayData, "numpy", reject_array_copy)
        patch.setattr(Array, "abs_arg_max", reject_legacy_peak)
        result = batch._process_batch()
    return result, copies


def test_live_batch_torch_peak_reduction_preserves_boundaries(
        torch_device, monkeypatch):
    rows = np.array([
        [99, 0, 0, 0, 0, 0, 0, 99],
        [99, 0, 0, 1, 3, -3, 2, 99],
        [99, 0, 0, 1, 2, np.nan, 3, 99],
        [99, 0, 0, 1, 2, 3, -4j, 99],
    ], dtype=np.complex64)

    with scheme.TorchScheme(torch_device):
        batch, templates, calls = _build_batch(rows, threshold=2.5)
        legacy_peaks = _legacy_torch_peaks(templates)
        (result, veto_info), copies = _run_without_scalar_sync(
            batch, monkeypatch
        )

    assert legacy_peaks[0][0] == 0
    assert legacy_peaks[1][0] == 1
    assert legacy_peaks[3][0] == 3
    assert legacy_peaks[0][1] == 0
    assert legacy_peaks[1][1] == 3
    assert legacy_peaks[3][1] == -4j

    assert calls == {"correlate": 1, "ifft": 1}
    assert batch.block_id == 1
    assert copies == [
        ((len(rows),), torch.int64, torch_device),
        ((len(rows),), torch.complex64, torch_device),
    ]
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


def test_live_batch_torch_peak_reduction_preserves_abort(
        torch_device, monkeypatch, caplog):
    rows = np.zeros((32, 8), dtype=np.complex64)
    rows[0, 4] = 3
    rows[1, 5] = 10
    rows[2:, 6] = 4

    with scheme.TorchScheme(torch_device):
        batch, templates, calls = _build_batch(
            rows, threshold=2, abort_threshold=5
        )
        sigmasq_calls = []
        for index, template in enumerate(templates):
            def ordered_sigmasq(_psd, template_index=index):
                sigmasq_calls.append(template_index)
                if template_index >= 2:
                    raise AssertionError(
                        "abort path evaluated a later normalization"
                    )
                return 1.0

            template.sigmasq = ordered_sigmasq

        monkeypatch.setattr(
            matchedfilter,
            "_torch_batch_peak_magnitudes",
            lambda *_args: pytest.fail(
                "abort path used the batch magnitude optimization"
            ),
        )
        with caplog.at_level(logging.INFO, logger="pycbc.filter.matchedfilter"):
            (result, veto_info), copies = _run_without_scalar_sync(
                batch, monkeypatch
            )

    assert result is False
    assert veto_info == []
    assert calls == {"correlate": 1, "ifft": 1}
    assert sigmasq_calls == [0, 1]
    assert batch.block_id == 1
    assert copies == [
        ((len(rows),), torch.int64, torch_device),
        ((len(rows),), torch.complex64, torch_device),
    ]
    assert caplog.messages == [
        "We are seeing some *really* high SNRs, let's assume they aren't "
        "signals and just give up"
    ]


def test_live_batch_torch_batch_magnitudes_match_scalar_contract():
    finite_peaks = [
        complex(-0.0, 0.0),
        3 + 4j,
        -3 - 4j,
        np.nextafter(np.float32(5), np.float32(0)) + 0j,
        np.nextafter(np.float32(5), np.float32(np.inf)) + 0j,
    ]
    special_components = [0.0, -0.0, np.inf, -np.inf, np.nan]
    special_peaks = [
        complex(real, imag)
        for real in special_components
        for imag in special_components
    ]
    peaks = np.array(finite_peaks + special_peaks, dtype=np.complex64)
    expected = np.fromiter(
        (abs(np.array([peak.item()])[0]) for peak in peaks),
        dtype=np.float64,
        count=len(peaks),
    )
    actual = matchedfilter._torch_batch_peak_magnitudes(peaks)

    np.testing.assert_array_equal(
        actual.view(np.uint64), expected.view(np.uint64)
    )
    assert matchedfilter._torch_batch_peak_magnitudes(peaks[None, :]) is None

    adversarial_peak = np.array(
        [np.complex64(0.020885564 + 4.8531j)], dtype=np.complex64
    )
    expected = abs(np.array([adversarial_peak[0].item()])[0])
    actual = matchedfilter._torch_batch_peak_magnitudes(adversarial_peak)[0]
    assert actual.view(np.uint64) == expected.view(np.uint64)


def test_live_batch_torch_off_axis_threshold_matches_scalar_path(
        torch_device, monkeypatch):
    peak = np.complex64(0.020885564 + 4.8531j)
    threshold = abs(np.array([peak.item()])[0])
    rows = np.zeros((32, 8), dtype=np.complex64)
    rows[0, 4] = peak

    with scheme.TorchScheme(torch_device):
        candidate, _templates, _calls = _build_batch(rows, threshold)
        candidate_result, candidate_veto = candidate._process_batch()

        reference, _templates, _calls = _build_batch(rows, threshold)
        with monkeypatch.context() as patch:
            patch.setattr(
                matchedfilter,
                "_torch_batch_peak_magnitudes",
                lambda _peaks: None,
            )
            reference_result, reference_veto = reference._process_batch()

    assert candidate_result.keys() == reference_result.keys()
    for key in candidate_result:
        np.testing.assert_array_equal(
            candidate_result[key], reference_result[key]
        )
    np.testing.assert_array_equal(candidate_result["template_id"], [10])
    assert len(candidate_veto) == len(reference_veto) == 1
    np.testing.assert_array_equal(candidate_veto[0][0], reference_veto[0][0])
    assert candidate_veto[0][1:3] == reference_veto[0][1:3]


def test_live_batch_torch_batch_magnitudes_are_scalar_exact():
    rng = np.random.default_rng(4971)
    components = rng.integers(
        0, 2**32, size=(8192, 2), dtype=np.uint32
    ).view(np.float32)
    components = components[np.isfinite(components).all(axis=1)]
    peaks = np.empty(len(components), dtype=np.complex64)
    peaks.real = components[:, 0]
    peaks.imag = components[:, 1]

    scalar_magnitudes = np.fromiter(
        (abs(np.array([peak.item()])[0]) for peak in peaks),
        dtype=np.float64,
        count=len(peaks),
    )
    batch_magnitudes = matchedfilter._torch_batch_peak_magnitudes(peaks)
    np.testing.assert_array_equal(
        batch_magnitudes.view(np.uint64),
        scalar_magnitudes.view(np.uint64),
    )


def test_live_batch_torch_batch_magnitudes_active_for_all_sizes(
        torch_device, monkeypatch):
    rows = np.zeros((31, 8), dtype=np.complex64)
    rows[:, 4] = 4
    called = []

    with scheme.TorchScheme(torch_device):
        batch, _templates, _calls = _build_batch(rows, threshold=3)
        orig_fn = matchedfilter._torch_batch_peak_magnitudes

        def record_call(peaks):
            called.append(len(peaks))
            return orig_fn(peaks)

        monkeypatch.setattr(
            matchedfilter,
            "_torch_batch_peak_magnitudes",
            record_call,
        )
        (result, _veto_info), _copies = _run_without_scalar_sync(
            batch, monkeypatch
        )

    assert called == [31]
    np.testing.assert_array_equal(result["template_id"], 10 + np.arange(31))


def test_live_batch_torch_batch_magnitude_none_keeps_scalar_fallback(
        torch_device, monkeypatch):
    rows = np.zeros((32, 8), dtype=np.complex64)
    selected = np.arange(32) % 4 == 0
    rows[:, 4] = np.where(selected, 4, 2)
    helper_calls = []

    with scheme.TorchScheme(torch_device):
        batch, _templates, _calls = _build_batch(rows, threshold=3)

        def decline_magnitudes(peaks):
            helper_calls.append(len(peaks))
            return None

        monkeypatch.setattr(
            matchedfilter, "_torch_batch_peak_magnitudes", decline_magnitudes
        )
        (result, veto_info), _copies = _run_without_scalar_sync(
            batch, monkeypatch
        )

    assert helper_calls == [32]
    selected_indices = np.flatnonzero(selected)
    np.testing.assert_array_equal(
        result["template_id"], 10 + selected_indices
    )
    assert all(info[0].dtype == np.dtype(np.complex128) for info in veto_info)


def test_live_batch_torch_batch_magnitudes_preserve_exception_order(
        torch_device, monkeypatch):
    class ExpectedSigmasqError(RuntimeError):
        pass

    rows = np.zeros((32, 8), dtype=np.complex64)
    rows[:, 4] = 4
    sigmasq_calls = []

    with scheme.TorchScheme(torch_device):
        batch, templates, _calls = _build_batch(rows, threshold=3)
        for index, template in enumerate(templates):
            def ordered_sigmasq(_psd, template_index=index):
                sigmasq_calls.append(template_index)
                if template_index == 5:
                    raise ExpectedSigmasqError
                return 1.0

            template.sigmasq = ordered_sigmasq

        with pytest.raises(ExpectedSigmasqError):
            _run_without_scalar_sync(batch, monkeypatch)

    assert sigmasq_calls == list(range(6))
    assert not any(
        hasattr(template, "dict_params") for template in templates
    )


def test_live_batch_torch_batch_threshold_at_measured_crossover(
        torch_device, monkeypatch):
    template_count = 32
    relative_indices = np.arange(template_count) % 4
    selected = (np.arange(template_count) % 3) == 0
    rows = np.zeros((template_count, 8), dtype=np.complex64)
    for index, relative_index in enumerate(relative_indices):
        if selected[index]:
            peak = 4 if index % 2 == 0 else -4j
        else:
            peak = 2
        rows[index, 3 + relative_index] = peak

    with scheme.TorchScheme(torch_device):
        batch, templates, calls = _build_batch(rows, threshold=3)
        sigmasq_calls = np.zeros(template_count, dtype=np.uint32)
        for index, template in enumerate(templates):
            def count_sigmasq(_psd, template_index=index):
                sigmasq_calls[template_index] += 1
                return 1.0

            template.sigmasq = count_sigmasq

        magnitude_calls = []
        original_magnitudes = matchedfilter._torch_batch_peak_magnitudes

        def count_magnitudes(peaks):
            magnitude_calls.append(np.array(peaks, copy=True))
            return original_magnitudes(peaks)

        monkeypatch.setattr(
            matchedfilter, "_torch_batch_peak_magnitudes", count_magnitudes
        )
        (result, veto_info), copies = _run_without_scalar_sync(
            batch, monkeypatch
        )

    selected_indices = np.flatnonzero(selected)
    assert calls == {"correlate": 1, "ifft": 1}
    assert len(magnitude_calls) == 1
    np.testing.assert_array_equal(sigmasq_calls, 1)
    np.testing.assert_array_equal(
        result["template_id"], 10 + selected_indices
    )
    np.testing.assert_array_equal(
        result["mass1"], 20 + selected_indices
    )
    np.testing.assert_array_equal(
        result["end_time"], 100 + relative_indices[selected]
    )
    np.testing.assert_array_equal(result["snr"], 4)
    expected_phases = np.where(selected_indices % 2 == 0, 0, -np.pi / 2)
    np.testing.assert_allclose(result["coa_phase"], expected_phases)
    np.testing.assert_array_equal(result["sigmasq"], 1)
    np.testing.assert_array_equal(
        [info[2] for info in veto_info],
        3 + relative_indices[selected],
    )
    assert copies == [
        ((template_count,), torch.int64, torch_device),
        ((template_count,), torch.complex64, torch_device),
    ]


@pytest.mark.parametrize("device", _available_torch_devices())
def test_live_batch_torch_peak_results_survive_workspace_reuse(device):
    rows = np.zeros((3, 8), dtype=np.complex64)
    rows[0, 4] = 3 + 1j
    rows[1, 5] = -4j
    rows[2, 6] = 5 - 2j

    with scheme.TorchScheme(device):
        output = Array(rows.reshape(-1))
        source = output._data.tensor
        source.requires_grad_(True)
        source_before = source.detach().clone()
        version_before = source._version
        indices, peaks = matchedfilter._torch_batch_peak_values(
            output, 3, 8, slice(3, 7)
        )
        source.detach().zero_()

    np.testing.assert_array_equal(indices, [1, 2, 3])
    np.testing.assert_array_equal(peaks, [3 + 1j, -4j, 5 - 2j])
    assert source._version == version_before + 1
    assert source.requires_grad
    np.testing.assert_array_equal(source_before.cpu().numpy(), rows.reshape(-1))


def test_live_batch_cpu_storage_keeps_legacy_peak_path(monkeypatch):
    rows = np.zeros((2, 8), dtype=np.complex64)
    rows[0, 4] = 3
    rows[1, 6] = -4j
    batch, _templates, calls = _build_batch(rows, threshold=2)

    assert matchedfilter._torch_batch_peak_values(
        batch.out_mem[batch.mids[0]], 2, 8, slice(3, 7)
    ) is None

    peak_calls = []
    original_peak = Array.abs_arg_max

    def count_legacy_peak(array):
        peak_calls.append(len(array))
        return original_peak(array)

    monkeypatch.setattr(Array, "abs_arg_max", count_legacy_peak)
    result, veto_info = batch._process_batch()

    assert calls == {"correlate": 1, "ifft": 1}
    assert peak_calls == [4, 4]
    np.testing.assert_array_equal(result["template_id"], [10, 11])
    np.testing.assert_array_equal(result["end_time"], [101, 103])
    np.testing.assert_allclose(result["snr"], [3, 4])
    np.testing.assert_allclose(result["coa_phase"], [0, -np.pi / 2])
    assert [info[2] for info in veto_info] == [4, 6]


def test_live_batch_power_matrices_init():
    template_count = 4
    template_size = 16
    with scheme.TorchScheme("cpu"):
        templates = []
        for index in range(template_count):
            arr = np.arange(template_size, dtype=np.complex64) * (index + 1)
            template = FrequencySeries(arr, delta_f=0.25)
            template.id = 10 + index
            template.params = np.array(
                [(20.0 + index,)], dtype=[("mass1", np.float32)]
            )[0]
            template.sigmasq = lambda _psd: 1.0
            templates.append(template)

        batch = matchedfilter.LiveBatchMatchedFilter(
            templates,
            snr_threshold=0.0,
            chisq_bins=0,
            sg_chisq=types.SimpleNamespace(),
        )

        mid = batch.mids[0]
        assert mid in batch.power_matrices
        pm = batch.power_matrices[mid]
        assert pm.shape == (template_count, template_size)
        assert pm.dtype == np.float32
        for index in range(template_count):
            expected_power = np.abs(templates[index].numpy()) ** 2
            np.testing.assert_allclose(pm[index], expected_power, rtol=1e-5)
        assert batch._psd_cache == {}


def test_live_batch_power_matrices_not_built_for_numpy():
    template = FrequencySeries(np.ones(16, dtype=np.complex64), delta_f=0.25)
    template.id = 10
    template.params = np.array(
        [(20.0,)], dtype=[("mass1", np.float32)]
    )[0]

    batch = matchedfilter.LiveBatchMatchedFilter(
        [template],
        snr_threshold=0.0,
        chisq_bins=0,
        sg_chisq=types.SimpleNamespace(),
    )

    assert batch.power_matrices == {}
    assert batch._psd_cache == {}


def test_live_batch_vectorized_sigmasq_and_caching(torch_device):
    template_count = 32
    template_size = 16
    rows = np.zeros((template_count, template_size), dtype=np.complex64)
    rows[:, 4] = 4.0

    templates = []
    for index in range(template_count):
        arr = np.ones(template_size, dtype=np.complex64) * (index + 1)
        template = FrequencySeries(arr, delta_f=0.25)
        template.id = 10 + index
        template.params = np.array(
            [(20.0 + index,)], dtype=[("mass1", np.float32)]
        )[0]
        templates.append(template)

    mid = (4.0, template_count)
    psd_vals1 = np.ones(template_size, dtype=np.float32) * 2.0
    psd1 = FrequencySeries(psd_vals1, delta_f=0.25)

    with scheme.TorchScheme(torch_device):
        output = Array(rows.reshape(-1))
        for index, template in enumerate(templates):
            template.out = output[index * template_size:(index + 1) * template_size]

        batch = object.__new__(matchedfilter.LiveBatchMatchedFilter)
        batch.block_id = 0
        batch.tgroups = [templates]
        batch.chunk_tsamples = [template_size]
        batch.mids = [mid]
        batch.out_mem = {mid: output}
        batch.corr = [types.SimpleNamespace(execute=lambda _data: None)]
        batch.ifts = {mid: types.SimpleNamespace(execute=lambda: None)}
        batch.snr_threshold = 0.0
        batch.snr_abort_threshold = None
        batch.power_matrices = {
            mid: np.stack(
                [(np.abs(t.numpy()) ** 2).astype(np.float32) for t in templates],
                axis=0,
            )
        }
        batch._psd_cache = {}
        batch.data = types.SimpleNamespace(
            overwhitened_data=lambda _df: types.SimpleNamespace(psd=psd1),
            trim_padding=1,
            blocksize=4,
            sample_rate=1,
            start_time=100.0,
        )

        result1, veto_info1 = batch._process_batch()

        assert mid in batch._psd_cache
        assert id(psd1) in batch._psd_cache[mid]
        cached_sigmasqs, cached_norms = batch._psd_cache[mid][id(psd1)]
        expected_inv_psd = (4.0 * 0.25) / psd_vals1
        expected_sigmasqs = batch.power_matrices[mid].dot(expected_inv_psd)
        expected_norms = 4.0 * 0.25 / np.sqrt(expected_sigmasqs)
        np.testing.assert_allclose(cached_sigmasqs, expected_sigmasqs, rtol=1e-5)
        np.testing.assert_allclose(cached_norms, expected_norms, rtol=1e-5)
        np.testing.assert_allclose(result1["sigmasq"], expected_sigmasqs)

        # Test cache hit on second run
        batch.block_id = 0
        result2, _ = batch._process_batch()
        np.testing.assert_allclose(result2["sigmasq"], expected_sigmasqs)

        # Test new PSD cache miss & populate
        psd_vals2 = np.ones(template_size, dtype=np.float32) * 8.0
        psd2 = FrequencySeries(psd_vals2, delta_f=0.25)
        batch.data = types.SimpleNamespace(
            overwhitened_data=lambda _df: types.SimpleNamespace(psd=psd2),
            trim_padding=1,
            blocksize=4,
            sample_rate=1,
            start_time=100.0,
        )
        batch.block_id = 0
        result3, _ = batch._process_batch()
        assert id(psd2) in batch._psd_cache[mid]
        expected_inv_psd2 = (4.0 * 0.25) / psd_vals2
        expected_sigmasqs2 = batch.power_matrices[mid].dot(expected_inv_psd2)
        np.testing.assert_allclose(result3["sigmasq"], expected_sigmasqs2)


def test_live_batch_vectorized_sigmasq_cache_bounding():
    template_count = 32
    template_size = 16
    rows = np.zeros((template_count, template_size), dtype=np.complex64)
    rows[:, 4] = 4.0
    templates = []
    for index in range(template_count):
        arr = np.ones(template_size, dtype=np.complex64)
        template = FrequencySeries(arr, delta_f=0.25)
        template.id = 10 + index
        template.params = np.array(
            [(20.0 + index,)], dtype=[("mass1", np.float32)]
        )[0]
        templates.append(template)

    mid = ("test_bound", template_count)

    with scheme.TorchScheme("cpu"):
        output = Array(rows.reshape(-1))
        for index, template in enumerate(templates):
            template.out = output[index * template_size:(index + 1) * template_size]

        batch = object.__new__(matchedfilter.LiveBatchMatchedFilter)
        batch.tgroups = [templates]
        batch.chunk_tsamples = [template_size]
        batch.mids = [mid]
        batch.out_mem = {mid: output}
        batch.corr = [types.SimpleNamespace(execute=lambda _data: None)]
        batch.ifts = {mid: types.SimpleNamespace(execute=lambda: None)}
        batch.snr_threshold = 0.0
        batch.snr_abort_threshold = None
        batch.power_matrices = {
            mid: np.stack(
                [(np.abs(t.numpy()) ** 2).astype(np.float32) for t in templates],
                axis=0,
            )
        }
        batch._psd_cache = {}

        psds = [
            FrequencySeries(np.ones(template_size, dtype=np.float32) * (i + 1.0), delta_f=0.25)
            for i in range(40)
        ]
        for psd in psds:
            batch.block_id = 0
            batch.data = types.SimpleNamespace(
                overwhitened_data=lambda _df, p=psd: types.SimpleNamespace(psd=p),
                trim_padding=1,
                blocksize=4,
                sample_rate=1,
                start_time=100.0,
            )
            batch._process_batch()

        assert len(batch._psd_cache[mid]) == 32


def test_live_batch_vectorized_sigmasq_fallback():
    template_count = 32
    template_size = 16
    rows = np.zeros((template_count, template_size), dtype=np.complex64)
    rows[:, 4] = 4.0
    sigmasq_calls = []

    templates = []
    for index in range(template_count):
        template = types.SimpleNamespace(
            delta_f=0.25,
            id=10 + index,
            params=np.array([(20.0 + index,)], dtype=[("mass1", np.float32)])[0],
            sigmasq=lambda _psd, idx=index: sigmasq_calls.append(idx) or 1.0,
        )
        templates.append(template)

    mid = ("test_fallback", template_count)

    with scheme.TorchScheme("cpu"):
        output = Array(rows.reshape(-1))
        for index, template in enumerate(templates):
            template.out = output[index * template_size:(index + 1) * template_size]

        batch = object.__new__(matchedfilter.LiveBatchMatchedFilter)
        batch.block_id = 0
        batch.tgroups = [templates]
        batch.chunk_tsamples = [template_size]
        batch.mids = [mid]
        batch.out_mem = {mid: output}
        batch.corr = [types.SimpleNamespace(execute=lambda _data: None)]
        batch.ifts = {mid: types.SimpleNamespace(execute=lambda: None)}
        batch.snr_threshold = 0.0
        batch.snr_abort_threshold = None
        # No power_matrices provided
        batch.data = types.SimpleNamespace(
            overwhitened_data=lambda _df: types.SimpleNamespace(psd=object()),
            trim_padding=1,
            blocksize=4,
            sample_rate=1,
            start_time=100.0,
        )

        result, veto_info = batch._process_batch()

        assert len(sigmasq_calls) == template_count
        np.testing.assert_array_equal(result["sigmasq"], 1.0)
        assert len(result["template_id"]) == template_count


@pytest.mark.parametrize("device", _available_torch_devices())
def test_torch_batch_peak_and_threshold_gpu_unit(device):
    from pycbc.filter import matchedfilter_torch

    template_count = 64
    seg_len = 128
    # Create zero background
    values = torch.zeros(
        (template_count, seg_len), dtype=torch.complex64, device=device
    )
    # Put sub-threshold peak in row 0
    values[0, 10] = 3.0 + 0j
    # Put above-threshold peak in row 5 and row 20
    values[5, 42] = 8.0 + 6.0j  # |val| = 10.0
    values[20, 99] = -12.0j      # |val| = 12.0

    norms = np.ones(template_count, dtype=np.float64)

    # 1. Threshold = 15.0 -> No triggers cross
    surv_idx, peak_idx, peak_val, aborted = (
        matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            values, norms, snr_threshold=15.0
        )
    )
    assert not aborted
    assert len(surv_idx) == 0
    assert len(peak_idx) == 0
    assert len(peak_val) == 0

    # 2. Threshold = 9.0 -> Rows 5 and 20 cross
    surv_idx, peak_idx, peak_val, aborted = (
        matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            values, norms, snr_threshold=9.0
        )
    )
    assert not aborted
    np.testing.assert_array_equal(surv_idx, [5, 20])
    np.testing.assert_array_equal(peak_idx, [42, 99])
    np.testing.assert_allclose(peak_val, [8.0 + 6.0j, -12.0j])

    # 3. Abort threshold = 11.0 -> Row 20 triggers abort
    surv_idx, peak_idx, peak_val, aborted = (
        matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
            values, norms, snr_threshold=9.0, snr_abort_threshold=11.0
        )
    )
    assert aborted
    assert len(surv_idx) == 0


@pytest.mark.parametrize("device", _available_torch_devices())
def test_live_batch_ondevice_peaks_integration(device, monkeypatch):
    monkeypatch.setenv("PYCBC_TORCH_ONDEVICE_PEAKS", "1")
    template_count = 32
    template_size = 8
    rows = np.zeros((template_count, template_size), dtype=np.complex64)
    # Row 3 has trigger at index 4 (relative offset 1 in seg [3:7])
    rows[3, 4] = 6.0 + 8.0j  # |val| = 10.0

    with scheme.TorchScheme(device):
        # Case A: High threshold (no triggers)
        batch_no_trig, _t, _c = _build_batch(rows, threshold=15.0)
        result_none, veto_none = batch_no_trig._process_batch()
        assert len(result_none["snr"]) == 0
        assert len(veto_none) == 0

        # Case B: Threshold = 8.0 (row 3 triggers)
        batch_trig, _t, _c = _build_batch(rows, threshold=8.0)
        result_trig, veto_trig = batch_trig._process_batch()
        np.testing.assert_array_equal(result_trig["template_id"], [13])
        np.testing.assert_allclose(result_trig["snr"], [10.0])
        np.testing.assert_array_equal(result_trig["end_time"], [101])
        assert len(veto_trig) == 1
        assert veto_trig[0][2] == 4


def test_live_batch_inference_mode_active_during_search_processing():
    observed = {}

    def record_corr_exec(_data):
        observed["corr_inference"] = torch.is_inference_mode_enabled()

    def record_ifft_exec():
        observed["ifft_inference"] = torch.is_inference_mode_enabled()

    rows = np.zeros((4, 8), dtype=np.complex64)
    with scheme.TorchScheme("cpu"):
        batch, _templates, _calls = _build_batch(rows, threshold=2.0)
        batch.corr = [types.SimpleNamespace(execute=record_corr_exec)]
        batch.ifts = {batch.mids[0]: types.SimpleNamespace(execute=record_ifft_exec)}
        result, veto = batch._process_batch()

    assert observed.get("corr_inference") is True
    assert observed.get("ifft_inference") is True


def test_live_batch_async_streams_flag_initialization(monkeypatch):
    tmplt = FrequencySeries(np.zeros(16, dtype=np.complex64), delta_f=1.0)
    monkeypatch.setenv("PYCBC_TORCH_ASYNC_STREAMS", "1")
    batch = matchedfilter.LiveBatchMatchedFilter(
        [tmplt],
        snr_threshold=5.0,
        chisq_bins="",
        sg_chisq=None,
    )
    assert batch.enable_async_streams is True

    monkeypatch.setenv("PYCBC_TORCH_ASYNC_STREAMS", "0")
    batch2 = matchedfilter.LiveBatchMatchedFilter(
        [tmplt],
        snr_threshold=5.0,
        chisq_bins="",
        sg_chisq=None,
    )
    assert batch2.enable_async_streams is False

    # Explicit constructor override
    batch3 = matchedfilter.LiveBatchMatchedFilter(
        [tmplt],
        snr_threshold=5.0,
        chisq_bins="",
        sg_chisq=None,
        enable_async_streams=True,
    )
    assert batch3.enable_async_streams is True


def test_live_batch_async_streams_double_buffering_pipeline(monkeypatch):
    class FakeStream:
        def __init__(self, device=None):
            self.device = device
            self.waited_events = []

        def wait_event(self, event):
            self.waited_events.append(event)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeEvent:
        def __init__(self):
            self.recorded = False

        def record(self, stream=None):
            self.recorded = True

    events_created = []
    streams_created = []

    def fake_stream_ctor(device=None):
        s = FakeStream(device)
        streams_created.append(s)
        return s

    def fake_event_ctor():
        e = FakeEvent()
        events_created.append(e)
        return e

    pinned_tensors = []

    def record_pin(tensor):
        pinned_tensors.append(tensor)
        return tensor

    monkeypatch.setattr(torch.cuda, "Stream", fake_stream_ctor)
    monkeypatch.setattr(torch.cuda, "Event", fake_event_ctor)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: s)
    monkeypatch.setattr(torch.Tensor, "pin_memory", record_pin)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)

    # Build 2-block batch
    rows1 = np.zeros((4, 8), dtype=np.complex64)
    rows2 = np.zeros((4, 8), dtype=np.complex64)
    rows1[0, 4] = 3.0
    rows2[1, 5] = 4.0

    with scheme.TorchScheme("cpu"):
        batch, templates1, calls = _build_batch(rows1, threshold=2.0)
        batch.enable_async_streams = True

        # Setup fake CUDA device tensor on out_mem
        cuda_device = torch.device("cuda:0")
        monkeypatch.setattr(
            torch.Tensor,
            "device",
            property(lambda self: cuda_device if getattr(self, "_is_fake_cuda", False) else torch.device("cpu")),
        )

        mid1 = batch.mids[0]
        mid2 = ("test2", 4)
        out1 = Array(rows1.reshape(-1))
        out1._data.tensor._is_fake_cuda = True
        out2 = Array(rows2.reshape(-1))
        out2._data.tensor._is_fake_cuda = True

        templates2 = []
        for index in range(4):
            params = np.array([(20.0 + index,)], dtype=[("mass1", np.float32)])[0]
            templates2.append(types.SimpleNamespace(
                delta_f=0.25, id=20 + index, params=params,
                out=out2[index * 8:(index + 1) * 8], sigmasq=lambda _psd: 1.0,
            ))

        batch.tgroups = [templates1, templates2]
        batch.chunk_tsamples = [8, 8]
        batch.mids = [mid1, mid2]
        batch.out_mem = {mid1: out1, mid2: out2}
        batch.corr = [
            types.SimpleNamespace(execute=lambda _data: None),
            types.SimpleNamespace(execute=lambda _data: None),
        ]
        batch.ifts = {
            mid1: types.SimpleNamespace(execute=lambda: None),
            mid2: types.SimpleNamespace(execute=lambda: None),
        }

        dummy_stilde = FrequencySeries(np.zeros(16, dtype=np.complex64), delta_f=0.25)
        dummy_stilde.psd = object()
        batch.data.overwhitened_data = lambda _delta_f: dummy_stilde

        # Process block 0 -> should prefetch block 1
        result0, veto0 = batch._process_batch()
        assert batch.block_id == 1
        assert len(streams_created) >= 2
        assert batch._async_prefetched is not None
        assert batch._async_prefetched[0] == 1
        assert len(pinned_tensors) >= 1

        # Process block 1 -> should consume prefetched block 1
        result1, veto1 = batch._process_batch()
        assert batch.block_id == 2
        assert batch._async_prefetched is None
