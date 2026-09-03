import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import FrequencySeries, TimeSeries  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402
from pycbc.waveform import utils as waveform_utils  # noqa: E402
from pycbc.waveform import waveform as waveform_module  # noqa: E402


@pytest.fixture(autouse=True)
def preserve_scheme():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    return device


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _reject_host_transfer(_self):
    raise AssertionError("Torch taper copied series data to the host")


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize(
    "series_kind,dtype",
    (
        ("time", np.float32),
        ("time", np.float64),
        ("frequency", np.complex64),
        ("frequency", np.complex128),
    ),
)
def test_kaiser_tapers_match_scipy_and_stay_on_device(
    monkeypatch, torch_device, side, series_kind, dtype
):
    if torch_device == "mps" and dtype in (np.float64, np.complex128):
        pytest.skip("Torch MPS does not support double-precision PyCBC arrays")

    values = np.linspace(0.25, 2.0, 64, dtype=np.float64)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 1j * values[::-1]
    values = values.astype(dtype)

    _activate_scheme(_scheme.CPUScheme())
    if series_kind == "time":
        series_type = TimeSeries
        series_params = {"delta_t": 0.125, "epoch": 100}
        taper = waveform_utils.td_taper
        start, end = 101.0, 102.0
    else:
        series_type = FrequencySeries
        series_params = {"delta_f": 0.5, "epoch": 100}
        taper = waveform_utils.fd_taper
        start, end = 4.0, 8.0

    reference_input = series_type(values, **series_params)
    expected = taper(reference_input, start, end, beta=5.5, side=side)

    _activate_scheme(_scheme.TorchScheme(torch_device))
    series = series_type(values, **series_params)
    original = series._data.tensor.clone()
    with monkeypatch.context() as patch:
        patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
        patch.setattr(
            waveform_utils.signal,
            "get_window",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Torch taper constructed a SciPy window")
            ),
        )
        actual = taper(series, start, end, beta=5.5, side=side)

    assert actual._data.tensor.device.type == torch_device
    assert actual.dtype == np.dtype(dtype)
    assert torch.equal(series._data.tensor, original)
    tolerance = 3e-6 if dtype in (np.float32, np.complex64) else 2e-14
    np.testing.assert_allclose(
        actual._data.tensor.detach().cpu().numpy(),
        expected.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


def test_pretaylorf2_native_pipeline_stays_on_device(
    monkeypatch, torch_device
):
    params = dict(
        approximant="PreTaylorF2",
        mass1=30.0,
        mass2=20.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=0.25,
        f_lower=20.0,
        f_final=100.0,
        final_taper=12.0,
        distance=400.0,
        inclination=0.4,
        coa_phase=0.3,
    )

    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(**params)
    reference_data = tuple(series.numpy().copy() for series in reference)
    reference_epochs = tuple(series.start_time for series in reference)
    reference_offsets = tuple(series.time_offset for series in reference)

    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(torch_device))
    with monkeypatch.context() as patch:
        patch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
        patch.setattr(
            waveform_utils.signal,
            "get_window",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("PreTaylorF2 constructed a SciPy window")
            ),
        )
        patch.setattr(
            waveform_module.lalsimulation,
            "SimInspiralChooseFDWaveform",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("native PreTaylorF2 reached lalsimulation")
            ),
        )
        actual = get_fd_waveform(**params)

    tolerance = 5e-4 if torch_device == "mps" else 2e-11
    for generated, expected, epoch, offset in zip(
        actual, reference_data, reference_epochs, reference_offsets
    ):
        assert generated._data.tensor.device.type == torch_device
        assert generated.start_time == epoch
        assert generated.time_offset == pytest.approx(offset)
        np.testing.assert_allclose(
            generated._data.tensor.detach().cpu().numpy(),
            expected,
            rtol=tolerance,
            atol=1e-32,
        )
