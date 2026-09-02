import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.eobnrv2_torch import (  # noqa: E402
    _MODES,
    _find_rom_directory,
    eobnrv2_native_supported,
)

_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_EOBNRV2_NATIVE",
)
_BASE_PARAMS = dict(
    mass1=40.0,
    mass2=20.0,
    delta_f=0.25,
    f_lower=20.0,
    distance=500.0,
    inclination=0.7,
    coa_phase=0.3,
)
_PARITY_PARAMS = (
    _BASE_PARAMS,
    {
        **_BASE_PARAMS,
        "mass1": 20.0,
        "mass2": 80.0,
        "delta_f": 0.5,
        "f_lower": 10.3,
        "f_final": 512.0,
        "f_ref": 0.0,
        "distance": 300.0,
        "inclination": 1.2,
        "coa_phase": -0.5,
        "long_asc_nodes": 0.4,
    },
    {
        **_BASE_PARAMS,
        "mass1": 68.0,
        "mass2": 6.0,
        "delta_f": 1.0,
        "f_lower": 15.0,
        "f_final": 2048.0,
        "f_ref": 4000.0,
        "inclination": 2.1,
        "coa_phase": 2.3,
    },
    {
        **_BASE_PARAMS,
        "mass1": 35.0,
        "mass2": 35.0,
        "inclination": 1.5,
    },
)
_SEQUENCE_POINTS = np.array(
    [20.0, 20.25, 23.75, 40.0, 80.0, 160.0, 400.0]
)


@pytest.fixture(autouse=True)
def preserve_process_state():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    old_environment = {name: os.environ.get(name) for name in _NATIVE_FLAGS}
    try:
        yield
    finally:
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _require_rom_data():
    try:
        _find_rom_directory()
    except FileNotFoundError:
        pytest.skip("EOBNRv2HM ROM data is not available on LAL_DATA_PATH")


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _prepare_backend(native, device, global_flag):
    os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1" if global_flag else "0"
    if native is None:
        os.environ.pop("PYCBC_EOBNRV2_NATIVE", None)
    else:
        os.environ["PYCBC_EOBNRV2_NATIVE"] = "1" if native else "0"
    _activate_scheme(
        _scheme.CPUScheme() if device is None else _scheme.TorchScheme(device)
    )


def _generate(
    params,
    *,
    native,
    device=None,
    global_flag=False,
    approximant="EOBNRv2_ROM",
):
    _prepare_backend(native, device, global_flag)
    return get_fd_waveform(approximant=approximant, **params)


def _generate_sequence(
    params,
    *,
    native,
    device=None,
    global_flag=False,
    approximant="EOBNRv2_ROM",
    sample_points=_SEQUENCE_POINTS,
):
    _prepare_backend(native, device, global_flag)
    return get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )


def _snapshot(series_pair):
    return tuple(series.numpy().copy() for series in series_pair)


def _assert_parity(reference, actual, tolerance):
    for expected, result in zip(reference, actual):
        result_array = result.numpy()
        assert len(result_array) == len(expected)
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        nonzero = expected != 0.0
        assert nonzero.any()
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < tolerance


@pytest.mark.parametrize("approximant", ["EOBNRv2_ROM", "EOBNRv2HM_ROM"])
@pytest.mark.parametrize("params", _PARITY_PARAMS)
def test_eobnrv2_torch_matches_lalsimulation(approximant, params):
    _require_rom_data()
    reference = _snapshot(
        _generate(params, native=False, approximant=approximant)
    )
    actual = _generate(
        params, native=True, device="cpu", approximant=approximant
    )

    assert all(series._data.tensor.device.type == "cpu" for series in actual)
    assert all(series._data.tensor.dtype == torch.complex128 for series in actual)
    assert all(float(series.epoch) == -1.0 / params["delta_f"] for series in actual)
    _assert_parity(reference, actual, tolerance=3.0e-11)


@pytest.mark.parametrize("approximant", ["EOBNRv2_ROM", "EOBNRv2HM_ROM"])
def test_eobnrv2_global_flag_avoids_lalsimulation(
    monkeypatch, approximant
):
    _require_rom_data()
    reference = _snapshot(
        _generate(_BASE_PARAMS, native=False, approximant=approximant)
    )

    import pycbc.waveform.waveform as waveform_module

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native EOBNRv2_ROM called lalsimulation")

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lalsimulation,
    )
    actual = _generate(
        _BASE_PARAMS,
        native=None,
        device="cpu",
        global_flag=True,
        approximant=approximant,
    )
    _assert_parity(reference, actual, tolerance=3.0e-11)


def test_eobnrv2hm_mode_subsets_are_canonical_and_additive():
    _require_rom_data()
    full = _generate(
        _BASE_PARAMS,
        native=True,
        device="cpu",
        approximant="EOBNRv2HM_ROM",
    )
    singles = [
        _generate(
            {**_BASE_PARAMS, "mode_array": [mode]},
            native=True,
            device="cpu",
            approximant="EOBNRv2HM_ROM",
        )
        for mode in _MODES
    ]
    dominant = _generate(
        _BASE_PARAMS,
        native=True,
        device="cpu",
        approximant="EOBNRv2_ROM",
    )
    reordered = _generate(
        {
            **_BASE_PARAMS,
            "mode_array": [(5, 5), (2, 2), (5, 5)],
        },
        native=True,
        device="cpu",
        approximant="EOBNRv2HM_ROM",
    )
    empty = _generate(
        {**_BASE_PARAMS, "mode_array": []},
        native=True,
        device="cpu",
        approximant="EOBNRv2HM_ROM",
    )

    for polarization in range(2):
        single_arrays = [
            waveform[polarization].numpy() for waveform in singles
        ]
        np.testing.assert_allclose(
            full[polarization].numpy(),
            sum(single_arrays),
            rtol=2.0e-14,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            singles[0][polarization].numpy(),
            dominant[polarization].numpy(),
        )
        np.testing.assert_allclose(
            reordered[polarization].numpy(),
            single_arrays[0] + single_arrays[-1],
            rtol=2.0e-14,
            atol=0.0,
        )
        assert not np.any(empty[polarization].numpy())


@pytest.mark.parametrize("approximant", ["EOBNRv2_ROM", "EOBNRv2HM_ROM"])
def test_eobnrv2_sequence_matches_lalsimulation_grid_samples(approximant):
    _require_rom_data()
    reference = _generate(
        _BASE_PARAMS, native=False, approximant=approximant
    )
    actual = _generate_sequence(
        _BASE_PARAMS,
        native=True,
        device="cpu",
        approximant=approximant,
    )
    indices = (_SEQUENCE_POINTS / _BASE_PARAMS["delta_f"]).astype(int)

    assert all(array._data.tensor.device.type == "cpu" for array in actual)
    assert all(array._data.tensor.dtype == torch.complex128 for array in actual)
    for expected_series, result in zip(reference, actual):
        expected = expected_series.numpy()[indices]
        relative_error = np.linalg.norm(result.numpy() - expected) / np.linalg.norm(
            expected
        )
        assert relative_error < 3.0e-11


def test_eobnrv2hm_sequence_subset_matches_native_regular_grid():
    _require_rom_data()
    params = {
        **_BASE_PARAMS,
        "mode_array": [(5, 5), (2, 2), (5, 5)],
    }
    regular = _generate(
        params,
        native=True,
        device="cpu",
        approximant="EOBNRv2HM_ROM",
    )
    sequence = _generate_sequence(
        params,
        native=True,
        device="cpu",
        approximant="EOBNRv2HM_ROM",
    )
    indices = (_SEQUENCE_POINTS / _BASE_PARAMS["delta_f"]).astype(int)
    for expected, actual in zip(regular, sequence):
        np.testing.assert_array_equal(
            actual.numpy(), expected.numpy()[indices]
        )


@pytest.mark.parametrize("approximant", ["EOBNRv2_ROM", "EOBNRv2HM_ROM"])
def test_eobnrv2_sequence_global_flag_avoids_lalsimulation(
    monkeypatch, approximant
):
    _require_rom_data()
    import pycbc.waveform.waveform as waveform_module

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native EOBNRv2 sequence called lalsimulation")

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    actual = _generate_sequence(
        _BASE_PARAMS,
        native=None,
        device="cpu",
        global_flag=True,
        approximant=approximant,
    )
    assert all(np.any(array.numpy()) for array in actual)


def test_eobnrv2_sequence_ignores_long_asc_nodes():
    _require_rom_data()
    reference = _generate_sequence(
        _BASE_PARAMS,
        native=True,
        device="cpu",
        approximant="EOBNRv2HM_ROM",
    )
    rotated = _generate_sequence(
        {**_BASE_PARAMS, "long_asc_nodes": 0.7},
        native=True,
        device="cpu",
        approximant="EOBNRv2HM_ROM",
    )
    for expected, actual in zip(reference, rotated):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_eobnrv2_sequence_avoids_host_transfer(monkeypatch):
    _require_rom_data()
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    def reject_host_transfer(_self):
        raise AssertionError("native EOBNRv2 sequence transferred to NumPy")

    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_EOBNRV2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array(_SEQUENCE_POINTS)
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    hp, hc = get_fd_waveform_sequence(
        approximant="EOBNRv2HM_ROM",
        sample_points=sample_points,
        **_BASE_PARAMS,
    )

    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"approximant": "EOBNRv2_ROM"}, True),
        ({"approximant": "EOBNRv2HM_ROM"}, True),
        ({"approximant": "EOBNRv2_ROM", "spin1z": 0.1}, False),
        ({"approximant": "EOBNRv2_ROM", "spin2x": 0.1}, False),
        ({"approximant": "EOBNRv2_ROM", "lambda1": 100.0}, False),
        ({"approximant": "EOBNRv2_ROM", "dchi3": 0.1}, False),
        ({"approximant": "EOBNRv2_ROM", "phase_order": 4}, False),
        ({"approximant": "EOBNRv2_ROM", "mode_array": [(2, 2)]}, False),
        ({"approximant": "EOBNRv2HM_ROM", "mode_array": [(2, 2)]}, True),
        ({"approximant": "EOBNRv2HM_ROM", "mode_array": []}, True),
        (
            {
                "approximant": "EOBNRv2HM_ROM",
                "mode_array": [(5, 5), (2, 2), (5, 5)],
            },
            True,
        ),
        ({"approximant": "EOBNRv2HM_ROM", "mode_array": [(2, -2)]}, False),
        ({"approximant": "EOBNRv2HM_ROM", "mode_array": [(3, 2)]}, False),
        ({"approximant": "EOBNRv2HM_ROM", "mode_array": [(2.5, 2)]}, False),
    ],
)
def test_eobnrv2_native_support_boundary(params, expected):
    assert eobnrv2_native_supported(params) is expected


@pytest.mark.parametrize(
    ("approximant", "extra_params"),
    [
        ("EOBNRv2_ROM", {"spin1z": 0.1}),
        ("EOBNRv2HM_ROM", {"mode_array": [(2, -2)]}),
    ],
)
def test_eobnrv2_unsupported_parameters_use_lal(
    monkeypatch, approximant, extra_params
):
    import pycbc.waveform.eobnrv2_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_params):
        raise AssertionError("unsupported parameters entered native path")

    def expected_lal(*_args, **_kwargs):
        raise RuntimeError("LAL fallback reached")

    monkeypatch.setattr(native_module, "eobnrv2_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        expected_lal,
    )
    with pytest.raises(RuntimeError, match="LAL fallback reached"):
        _generate(
            {**_BASE_PARAMS, **extra_params},
            native=True,
            device="cpu",
            approximant=approximant,
        )


def test_eobnrv2_unsupported_sequence_parameters_use_lal(monkeypatch):
    import pycbc.waveform.eobnrv2_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_params):
        raise AssertionError("unsupported parameters entered native path")

    def expected_lal(*_args, **_kwargs):
        raise RuntimeError("LAL sequence fallback reached")

    monkeypatch.setattr(
        native_module, "eobnrv2_fd_sequence_torch", unexpected_native
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        expected_lal,
    )
    with pytest.raises(RuntimeError, match="LAL sequence fallback reached"):
        _generate_sequence(
            {**_BASE_PARAMS, "spin1z": 0.1},
            native=True,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("sample_points", "message"),
    [
        ([], "non-empty vector"),
        ([20.0, 20.0], "strictly increasing"),
        ([20.0, 0.0], "positive"),
        ([20.0, np.nan], "finite"),
    ],
)
def test_eobnrv2_sequence_validates_sample_points(sample_points, message):
    _require_rom_data()
    with pytest.raises(ValueError, match=message):
        _generate_sequence(
            _BASE_PARAMS,
            native=True,
            device="cpu",
            sample_points=sample_points,
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
@pytest.mark.parametrize("approximant", ["EOBNRv2_ROM", "EOBNRv2HM_ROM"])
def test_eobnrv2_mps_residency_and_parity(approximant):
    _require_rom_data()
    reference = _snapshot(
        _generate(_BASE_PARAMS, native=False, approximant=approximant)
    )
    actual = _generate(
        _BASE_PARAMS, native=True, device="mps", approximant=approximant
    )

    assert all(series._data.tensor.device.type == "mps" for series in actual)
    assert all(series._data.tensor.dtype == torch.complex64 for series in actual)
    _assert_parity(reference, actual, tolerance=5.0e-3)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
@pytest.mark.parametrize("approximant", ["EOBNRv2_ROM", "EOBNRv2HM_ROM"])
def test_eobnrv2_sequence_mps_residency_and_parity(approximant):
    _require_rom_data()
    reference = _snapshot(
        _generate_sequence(
            _BASE_PARAMS,
            native=True,
            device="cpu",
            approximant=approximant,
        )
    )
    actual = _generate_sequence(
        _BASE_PARAMS,
        native=True,
        device="mps",
        approximant=approximant,
    )

    assert all(array._data.tensor.device.type == "mps" for array in actual)
    assert all(array._data.tensor.dtype == torch.complex64 for array in actual)
    _assert_parity(reference, actual, tolerance=5.0e-3)
