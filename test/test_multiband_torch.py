import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import zeros
from pycbc.waveform import get_fd_waveform
from pycbc.waveform import multiband


_MULTIBAND_PARAMS = dict(
    approximant="multiband",
    base_approximant="TaylorF2",
    mass1=20.0,
    mass2=15.0,
    spin1z=0.1,
    spin2z=-0.05,
    delta_f=0.25,
    f_lower=20.0,
    f_final=128.0,
    bands=[64.0],
    lengths=[1.0],
    overlap=8.0,
    distance=100.0,
)


@pytest.fixture
def preserve_scheme():
    """Restore the process-wide PyCBC scheme singleton after a test."""
    old_scheme = scheme.mgr.state
    old_single = scheme.Scheme._single
    try:
        yield
    finally:
        scheme.mgr.state = old_scheme
        scheme.Scheme._single = old_single


def _activate_scheme(state):
    scheme.Scheme._single = None
    scheme.mgr.state = state


def _reject_numpy_hanning(*_args, **_kwargs):
    raise AssertionError("multiband Torch stitching used numpy.hanning")


@pytest.mark.parametrize("length", (0, 1, 8, 9))
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_torch_hann_window_matches_numpy(
    length, dtype, monkeypatch, preserve_scheme
):
    torch = pytest.importorskip("torch")
    expected = np.hanning(length).astype(np.empty(0, dtype=dtype).real.dtype)

    _activate_scheme(scheme.TorchScheme("cpu"))
    reference = zeros(4, dtype=dtype)
    monkeypatch.setattr(multiband.numpy, "hanning", _reject_numpy_hanning)
    actual = multiband._hann_window(length, reference)

    tensor = actual._data.tensor
    assert tensor.device.type == "cpu"
    torch.testing.assert_close(tensor, torch.as_tensor(expected))


@pytest.mark.parametrize("device", ("cpu", "cuda", "mps"))
def test_public_multiband_stays_on_torch_device(
    device, monkeypatch, preserve_scheme
):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")

    from pycbc.types.array_torch import TorchArrayData

    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")

    _activate_scheme(scheme.CPUScheme())
    reference = get_fd_waveform(**_MULTIBAND_PARAMS)
    reference_arrays = tuple(series.numpy() for series in reference)

    _activate_scheme(scheme.TorchScheme(device))

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("multiband Torch stitching copied data to host")

    with monkeypatch.context() as patch:
        patch.setattr(
            multiband.numpy, "hanning", _reject_numpy_hanning
        )
        patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
        actual = get_fd_waveform(**_MULTIBAND_PARAMS)

    tolerance = 1.0e-4 if device == "mps" else 2.0e-6
    for expected, series in zip(reference_arrays, actual):
        tensor = series._data.tensor
        assert tensor.device.type == device
        values = tensor.detach().cpu().numpy()
        np.testing.assert_array_equal(values == 0.0, expected == 0.0)

        nonzero = np.abs(expected) > 0.0
        expected_nonzero = expected[nonzero].astype(np.complex128)
        values_nonzero = values[nonzero].astype(np.complex128)
        relative_error = np.linalg.norm(
            values_nonzero - expected_nonzero
        ) / np.linalg.norm(expected_nonzero)
        assert relative_error < tolerance
