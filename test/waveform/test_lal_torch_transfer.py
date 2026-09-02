"""Tests for ownership-safe regular LAL-to-Torch waveform transfers."""

from contextlib import contextmanager
import gc
import weakref

import numpy as np
import pytest


torch = pytest.importorskip("torch")


@contextmanager
def _active_state(state):
    from pycbc import scheme

    old_state = scheme.mgr.state
    old_lock = scheme.mgr._lock
    scheme.mgr._lock = False
    scheme.mgr.state = state
    try:
        yield
    finally:
        scheme.mgr._lock = False
        scheme.mgr.state = old_state
        scheme.mgr._lock = old_lock


def _torch_state(device):
    from pycbc import scheme

    return scheme.TorchScheme(device)


def _cpu_state():
    from pycbc import scheme

    state = object.__new__(scheme.CPUScheme)
    state.num_threads = 1
    state._libgomp = None
    state._owns_singleton = False
    return state


def _source_values():
    return np.array(
        [complex(-0.0, 0.0), 1.0 + 2.0j, -3.0 - 4.0j],
        dtype=np.complex128,
    )


def test_regular_cpu_transfer_aliases_pointer_and_retains_lal_owner():
    from pycbc.types import FrequencySeries
    from pycbc.waveform import waveform

    def build_series():
        owner = _source_values()
        source = owner[:]
        pointer = int(source.__array_interface__["data"][0])
        source_ref = weakref.ref(source)
        owner_ref = weakref.ref(owner)
        with _active_state(_torch_state("cpu")):
            series = waveform._series_from_lal_output(
                source,
                FrequencySeries,
                "delta_f",
                0.25,
                0,
                "TaylorF2",
            )
        return series, pointer, source_ref, owner_ref

    series, pointer, source_ref, owner_ref = build_series()
    tensor = series._data.tensor

    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.complex128
    assert tensor.data_ptr() == pointer
    assert source_ref() is not None
    assert owner_ref() is not None
    assert tensor.detach().numpy().tobytes(order="C") == _source_values().tobytes(
        order="C"
    )

    source_ref()[1] = 7.0 + 8.0j
    assert tensor[1].item() == 7.0 + 8.0j
    tensor[2] = -9.0 + 10.0j
    assert source_ref()[2] == -9.0 + 10.0j

    del tensor
    del series
    gc.collect()
    assert source_ref() is None
    assert owner_ref() is None


def test_standard_cpu_series_keeps_established_copy_semantics():
    from pycbc.types import FrequencySeries
    from pycbc.waveform import waveform

    owner = _source_values()
    source = owner[:]
    expected = source.tobytes(order="C")
    source_pointer = int(source.__array_interface__["data"][0])

    with _active_state(_cpu_state()):
        series = waveform._series_from_lal_output(
            source,
            FrequencySeries,
            "delta_f",
            0.25,
            0,
            "TaylorF2",
        )

    assert isinstance(series._data, np.ndarray)
    assert int(series._data.__array_interface__["data"][0]) != source_pointer
    assert not np.shares_memory(series._data, source)
    assert series._data.tobytes(order="C") == expected
    source[:] = 99.0 + 101.0j
    assert series._data.tobytes(order="C") == expected


def test_ineligible_torch_input_falls_back_to_copying_series_path():
    from pycbc.types import FrequencySeries
    from pycbc.waveform import waveform

    owner = np.arange(12, dtype=np.float64)
    source = owner[::2]
    expected = source.copy()

    with _active_state(_torch_state("cpu")):
        series = waveform._series_from_lal_output(
            source,
            FrequencySeries,
            "delta_f",
            0.25,
            0,
            "TaylorF2",
        )

    assert series._data.tensor.device.type == "cpu"
    assert series._data.tensor.data_ptr() != int(
        source.__array_interface__["data"][0]
    )
    np.testing.assert_array_equal(series._data.tensor.numpy(), expected)
    source[:] = -1.0
    np.testing.assert_array_equal(series._data.tensor.numpy(), expected)


def test_unqualified_approximant_keeps_established_copying_path():
    from pycbc.types import FrequencySeries
    from pycbc.waveform import waveform

    source = _source_values()
    expected = source.copy()

    with _active_state(_torch_state("cpu")):
        series = waveform._series_from_lal_output(
            source,
            FrequencySeries,
            "delta_f",
            0.25,
            0,
            "UnprofiledApproximant",
        )

    assert series._data.tensor.data_ptr() != int(
        source.__array_interface__["data"][0]
    )
    assert series._data.tensor.numpy().tobytes(order="C") == expected.tobytes(
        order="C"
    )
    source[:] = 99.0 + 101.0j
    assert series._data.tensor.numpy().tobytes(order="C") == expected.tobytes(
        order="C"
    )


def _read_only_array():
    values = np.arange(4, dtype=np.float64)
    values.flags.writeable = False
    return values


def _unaligned_array():
    storage = np.zeros(33, dtype=np.uint8)
    return np.ndarray((4,), dtype=np.float64, buffer=storage, offset=1)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: [1.0, 2.0],
        lambda: np.array(1.0, dtype=np.float64),
        lambda: np.array([], dtype=np.float64),
        lambda: np.ones((2, 2), dtype=np.float64),
        lambda: np.arange(8, dtype=np.float64)[::2],
        lambda: np.arange(4, dtype=np.int64),
        _read_only_array,
        lambda: np.arange(4, dtype=np.float64).astype(">f8"),
        _unaligned_array,
    ),
)
def test_transfer_eligibility_rejects_unsafe_input_without_torch_call(factory):
    from pycbc.waveform import waveform

    class RejectTorch:
        @staticmethod
        def from_numpy(_values):
            raise AssertionError("ineligible input reached torch.from_numpy")

    assert waveform._torch_lal_array_data(
        factory(),
        state=_torch_state("cpu"),
        torch_module=RejectTorch,
        array_data_type=object,
    ) is None


@pytest.mark.parametrize("device", ("cpu", "mps"))
def test_transfer_rejects_non_torch_or_unsupported_torch_scheme(device):
    from pycbc import scheme
    from pycbc.waveform import waveform

    class RejectTorch:
        @staticmethod
        def from_numpy(_values):
            raise AssertionError("unsupported scheme reached torch.from_numpy")

    if device == "cpu":
        state = _cpu_state()
    else:
        state = object.__new__(scheme.TorchScheme)
        state.torch_device = torch.device("mps")
    assert waveform._torch_lal_array_data(
        _source_values(),
        state=state,
        torch_module=RejectTorch,
        array_data_type=object,
    ) is None


def test_cuda_transfer_requests_exactly_one_blocking_host_copy():
    from pycbc import scheme
    from pycbc.waveform import waveform

    calls = []

    class FakeDevice:
        type = "cuda"

    class FakeTarget:
        def __init__(self, values):
            self.values = values
            self.device = FakeDevice()

    class FakeHost:
        dtype = "torch.complex128"

        def __init__(self, values):
            self.values = values

        def to(self, **kwargs):
            calls.append(kwargs)
            return FakeTarget(self.values.copy())

    class FakeTorch:
        @staticmethod
        def from_numpy(values):
            return FakeHost(values)

    class FakeArrayData:
        def __init__(self, tensor):
            self.tensor = tensor

    state = object.__new__(scheme.TorchScheme)
    state.torch_device = FakeDevice()
    wrapped = waveform._torch_lal_array_data(
        _source_values(),
        state=state,
        torch_module=FakeTorch,
        array_data_type=FakeArrayData,
    )

    assert wrapped.tensor.device.type == "cuda"
    assert len(calls) == 1
    assert calls[0] == {
        "device": state.torch_device,
        "dtype": "torch.complex128",
        "non_blocking": False,
        "copy": True,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_transfer_is_bitwise_and_releases_host_after_one_copy():
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import waveform

    calls = []
    from_numpy_calls = []

    class CountingHost:
        def __init__(self, tensor):
            self.tensor = tensor
            self.dtype = tensor.dtype

        def to(self, **kwargs):
            calls.append(kwargs)
            return self.tensor.to(**kwargs)

    class CountingTorch:
        @staticmethod
        def from_numpy(values):
            from_numpy_calls.append(None)
            return CountingHost(torch.from_numpy(values))

    def build_transfer():
        owner = _source_values()
        source = owner[:]
        expected = source.tobytes(order="C")
        source_ref = weakref.ref(source)
        owner_ref = weakref.ref(owner)
        wrapped = waveform._torch_lal_array_data(
            source,
            state=_torch_state("cuda"),
            torch_module=CountingTorch,
            array_data_type=TorchArrayData,
        )
        return wrapped, expected, source_ref, owner_ref

    wrapped, expected, source_ref, owner_ref = build_transfer()
    torch.cuda.synchronize()
    gc.collect()

    assert from_numpy_calls == [None]
    assert len(calls) == 1
    assert calls[0]["device"].type == "cuda"
    assert calls[0]["dtype"] == torch.complex128
    assert calls[0]["non_blocking"] is False
    assert calls[0]["copy"] is True
    assert wrapped.tensor.device.type == "cuda"
    assert wrapped.tensor.cpu().numpy().tobytes(order="C") == expected
    assert source_ref() is None
    assert owner_ref() is None


_PUBLIC_CASES = (
    (
        "TaylorF2",
        "fd",
        {
            "mass1": 10.0,
            "mass2": 9.0,
            "spin1z": 0.2,
            "spin2z": -0.1,
            "delta_f": 4.0,
            "f_lower": 30.0,
            "f_final": 96.0,
            "distance": 500.0,
        },
    ),
    (
        "IMRPhenomD",
        "fd",
        {
            "mass1": 40.0,
            "mass2": 30.0,
            "spin1z": 0.2,
            "spin2z": -0.1,
            "delta_f": 4.0,
            "f_lower": 20.0,
            "f_final": 128.0,
            "f_ref": 20.0,
            "distance": 500.0,
        },
    ),
    (
        "IMRPhenomXPHM",
        "fd",
        {
            "mass1": 40.0,
            "mass2": 20.0,
            "spin1x": 0.2,
            "spin1y": -0.1,
            "spin1z": 0.3,
            "spin2x": -0.1,
            "spin2y": 0.05,
            "spin2z": -0.2,
            "delta_f": 4.0,
            "f_lower": 20.0,
            "f_final": 128.0,
            "f_ref": 20.0,
            "inclination": 0.8,
            "distance": 500.0,
        },
    ),
    (
        "TaylorT4",
        "td",
        {
            "mass1": 30.0,
            "mass2": 25.0,
            "delta_t": 1.0 / 1024.0,
            "f_lower": 40.0,
            "f_ref": 40.0,
            "distance": 500.0,
        },
    ),
)


@pytest.mark.parametrize("approximant,interface,parameters", _PUBLIC_CASES)
def test_regular_lal_torch_cpu_waveform_is_bitwise_standard_cpu(
    monkeypatch,
    approximant,
    interface,
    parameters,
):
    from pycbc.waveform import get_fd_waveform, get_td_waveform
    from pycbc.waveform import waveform

    if not waveform._lalsimulation_available:
        pytest.skip("lalsimulation is unavailable")

    for name in (
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
        f"PYCBC_{approximant.upper()}_NATIVE",
    ):
        monkeypatch.setenv(name, "0")

    generator = get_fd_waveform if interface == "fd" else get_td_waveform
    with _active_state(_cpu_state()):
        reference = generator(approximant=approximant, **parameters)
        reference_values = tuple(series.numpy().copy() for series in reference)
        reference_delta = tuple(
            series.delta_f if interface == "fd" else series.delta_t
            for series in reference
        )
        reference_epoch = tuple(
            float(series.epoch if interface == "fd" else series.start_time)
            for series in reference
        )

    with _active_state(_torch_state("cpu")):
        candidate = generator(approximant=approximant, **parameters)
        candidate_values = tuple(
            series._data.tensor.detach().numpy().copy() for series in candidate
        )
        candidate_delta = tuple(
            series.delta_f if interface == "fd" else series.delta_t
            for series in candidate
        )
        candidate_epoch = tuple(
            float(series.epoch if interface == "fd" else series.start_time)
            for series in candidate
        )

    assert candidate_delta == reference_delta
    assert candidate_epoch == reference_epoch
    for actual, expected in zip(candidate_values, reference_values):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes(order="C") == expected.tobytes(order="C")
